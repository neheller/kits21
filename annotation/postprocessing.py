import sys
from pathlib import Path
import json

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
import torch
import torch.nn.functional
from scipy import signal
from skimage import measure
import cv2

#pylint: disable=no-member


def load_json(json_path):
    with json_path.open() as f:
        return json.loads(f.read())


def write_json(json_path, data):
    with json_path.open("w") as f:
        return f.write(json.dumps(data, indent=2))


def get_containing_box(dln, shape):
    annotated_frames = set([])
    maxs = [0, 0]
    mins = [np.inf, np.inf]
    max_sz = 0
    for ann in dln["annotations"]:
        annotated_frames.add(ann["frame"])
        for pt in ann["spatial_payload"]:
            if pt[0] > maxs[0]:
                maxs[0] = pt[0]
            if pt[1] > maxs[1]:
                maxs[1] = pt[1]
            if pt[0] < mins[0]:
                mins[0] = pt[0]
            if pt[1] < mins[1]:
                mins[1] = pt[1]
        if ann["line_size"] > max_sz:
            max_sz = ann["line_size"]

    afrms = sorted(list(annotated_frames))
    last = afrms[0]
    min_step = np.inf
    for afrm in afrms[1:]:
        if afrm - last < min_step:
            min_step = afrm - last
        last = afrm

    return {
        "xmin": max(0, int(np.floor(mins[0] - max_sz))),
        "xmax": min(shape[2], int(np.ceil(maxs[0] + max_sz))),
        "ymin": max(0, int(np.floor(mins[1] - max_sz))),
        "ymax": min(shape[1], int(np.ceil(maxs[1] + max_sz))),
        "zmin": max(0, min(afrms) - min_step),
        "zmax": min(shape[0], max(afrms) + min_step),
        "step": min_step,
        "xdim": shape[2],
        "ydim": shape[1],
        "zdim": shape[0]
    }


def get_cropped_scan(cbox, img_nib):
    return img_nib.get_fdata()[
        cbox["zmin"]:cbox["zmax"] + 1,
        cbox["ymin"]:cbox["ymax"] + 1,
        cbox["xmin"]:cbox["xmax"] + 1
    ]


def generate_cropped_drawing_interior(cbox, dln):
    ret = np.zeros((
        cbox["zmax"] - cbox["zmin"] + 1,
        cbox["ymax"] - cbox["ymin"] + 1,
        cbox["xmax"] - cbox["xmin"] + 1
    ), dtype=np.int)

    for i in range(ret.shape[0]):
        with Image.new("L", (ret.shape[2]*10, ret.shape[1]*10)) as im:
            draw = ImageDraw.Draw(im)
            drew = False
            for stroke in dln["annotations"]:
                if stroke["deprecated"]:
                    continue
                if i + cbox["zmin"] == stroke["frame"]:
                    drew = True
                    draw.line(
                        [
                            (
                                int(round((x[0] - cbox["xmin"])*10)),
                                int(round((x[1] - cbox["ymin"])*10))
                            ) 
                            for x in stroke["spatial_payload"]
                        ],
                        fill=128,
                        width=int(round(stroke["line_size"]*10))+4,
                        joint="curve"
                    )
                    srt = stroke["spatial_payload"][0]
                    draw.ellipse(
                        [
                            (
                                int(round((srt[0] - cbox["xmin"] - stroke["line_size"]/2)*10))-2,
                                int(round((srt[1] - cbox["ymin"] - stroke["line_size"]/2)*10))-2
                            ),
                            (
                                int(round((srt[0] - cbox["xmin"] + stroke["line_size"]/2)*10))+2,
                                int(round((srt[1] - cbox["ymin"] + stroke["line_size"]/2)*10))+2
                            )
                        ],
                        fill=128
                    )
                    end = stroke["spatial_payload"][-1]
                    draw.ellipse(
                        [
                            (
                                int(round((end[0] - cbox["xmin"] - stroke["line_size"]/2)*10))-2,
                                int(round((end[1] - cbox["ymin"] - stroke["line_size"]/2)*10))-2
                            ),
                            (
                                int(round((end[0] - cbox["xmin"] + stroke["line_size"]/2)*10))+2,
                                int(round((end[1] - cbox["ymin"] + stroke["line_size"]/2)*10))+2
                            )
                        ],
                        fill=128
                    )
            if drew:
                rszd = im.resize((ret.shape[2], ret.shape[1]), Image.BILINEAR)
                ImageDraw.floodfill(rszd, (0,0), 128, thresh=63.5)
                ret[i,:,:] = np.less(np.array(rszd), 63.9).astype(np.int)

    return ret


def get_contour(bin_seg):
    if bin_seg is None:
        return None
    contours, hierarchy = cv2.findContours(bin_seg.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def distance(p1, p2):
    return (p1[0][0] - p2[0][0])*(p1[0][0] - p2[0][0]) + (p1[0][1] - p2[0][1])*(p1[0][1] - p2[0][1])

            
def find_nearest_neighbors_slow(lg_cntr, sm_cntr, mean_dsp):
    matches = np.zeros_like(lg_cntr)
    for i in range(lg_cntr.shape[0]):
        minj = None
        mind = np.inf
        ipt = lg_cntr[i] + np.array([mean_dsp])
        for j in range(sm_cntr.shape[0]):
            dst = distance(ipt, sm_cntr[j])
            if dst < mind:
                minj = j
                mind = dst
        matches[i] = sm_cntr[minj]
    return matches


def find_nearest_neighbors_slow_v2(lg_cntr, sm_cntr, mean_dsp):
    matches = np.zeros_like(lg_cntr)
    step = sm_cntr.shape[0]/lg_cntr.shape[0]
    mini = None
    mind = np.inf
    # ipt = lg_cntr[i] + np.array([mean_dsp])
    for i in range(lg_cntr.shape[0]):
        candidate_matches = np.zeros_like(lg_cntr)
        offset = i*step
        for j in range(lg_cntr.shape[0]):
            candidate_matches[j] = sm_cntr[int(np.round(offset + j*step)) % sm_cntr.shape[0]]

        dist = np.square(lg_cntr - candidate_matches).sum()
        if dist < mind:
            mini = i
            matches = candidate_matches.copy()
            mind = dist
            
    return matches


def find_nearest_neighbors(lg_cntr, sm_cntr, mean_dsp):
    print(mean_dsp)


def draw_filled_contour(ind, bef_i, aft_i, drw_c, bef_bin, aft_bin, float_contour):
    blown_up = np.zeros((drw_c.shape[1]*10, drw_c.shape[2]*10), dtype=np.uint8)
    points = np.round(float_contour*10).astype(np.int32) + 1
    cv2.fillPoly(blown_up, pts=[points], color=128)
    drw_c[ind,:,:] = np.logical_or(
        drw_c[ind,:,:],
        np.logical_or(
            np.greater(cv2.resize(blown_up, (drw_c.shape[2], drw_c.shape[1]), cv2.INTER_LINEAR), 32),
            np.multiply(bef_bin, aft_bin)
        )
    )


def interpolate_simple_association(bef_bin, aft_bin, drw_c, bef_i, aft_i, bef_cnt, aft_cnt, step):
    # cnt <- center
    # cntr <- contour
    bef_cntr = get_contour(bef_bin)
    aft_cntr = get_contour(aft_bin)
    debug = False
    if bef_cntr is None:
        debug = True
        start = bef_i
        inc = 1
        ref = bef_cntr
        bef_cntr = np.array([
            [bef_cnt]
        ])
        bef_bin = np.zeros_like(aft_bin)
    elif aft_cntr is None:
        debug = True
        start = aft_i
        inc = -1
        ref = aft_cntr
        aft_cntr = np.array([
            [aft_cnt]
        ])
        aft_bin = np.zeros_like(bef_bin)
    if bef_cntr.shape[0] > aft_cntr.shape[0]:
        start = bef_i
        inc = 1
        ref = bef_cntr
        matches = find_nearest_neighbors_slow_v2(bef_cntr, aft_cntr, [aft_cnt[0] - bef_cnt[0], aft_cnt[1] - bef_cnt[1]])
    else:
        start = aft_i
        inc = -1
        ref = aft_cntr
        matches = find_nearest_neighbors_slow_v2(aft_cntr, bef_cntr, [bef_cnt[0] - aft_cnt[0], bef_cnt[1] - aft_cnt[1]])

    for i in range(1, aft_i - bef_i):
        # if debug:
        #     print(ref, matches, i/step*matches + (step - i)/step*ref)

        draw_filled_contour(
            start + i*inc, bef_i, aft_i,
            drw_c, bef_bin, aft_bin,
            i/step*matches + (step - i)/step*ref
        )



def interpolate_step(bef_i, aft_i, drw_c, step):
    # Label connected components in each
    bef_lbl = measure.label(drw_c[bef_i, :, :], background=0)
    aft_lbl = measure.label(drw_c[aft_i, :, :], background=0)

    # Associate connected components based on proximity and overlap
    num_bef = np.max(bef_lbl)
    num_aft = np.max(aft_lbl)

    aft_cvg = [False for _ in range(num_aft)]

    bef_to_aft = {}
    aft_to_bef = {}

    # Iterate over all pairs of blobs
    for i in range(1, num_bef+1):
        bef_bin = np.equal(bef_lbl, i).astype(np.int)
        bef_cnt_x, bef_cnt_y = np.argwhere(bef_bin == 1).sum(0)/bef_bin.sum()
        bef_covered = False
        istr = "{}".format(i)
        for j in range(1, num_aft+1):
            aft_bin = np.equal(aft_lbl, j).astype(np.int)

            # Get size of overlap
            ovr_sz = np.multiply(bef_bin, aft_bin).sum()

            # Get metrics describing blob proximity
            aft_cnt_x, aft_cnt_y = np.argwhere(aft_bin == 1).sum(0)/aft_bin.sum()
            cnt_dsp = [aft_cnt_y - bef_cnt_y, aft_cnt_x - bef_cnt_x]
            cnt_dst_sq = cnt_dsp[0]**2 + cnt_dsp[1]**2

            if ovr_sz > 0 or cnt_dst_sq < 5**2:
                jstr = "{}".format(j)
                if istr not in bef_to_aft:
                    bef_to_aft[istr] = []
                bef_to_aft[istr] += [{
                    "ind": j,
                    "ovr_sz": int(ovr_sz),
                    "cnt_dst_sq": cnt_dst_sq
                }]
                if jstr not in aft_to_bef:
                    aft_to_bef[jstr] = []
                aft_to_bef[jstr] += [{
                    "ind": i,
                    "ovr_sz": int(ovr_sz),
                    "cnt_dst_sq": cnt_dst_sq
                }]
                bef_covered = True
                aft_cvg[j-1] = True
                interpolate_simple_association(
                    bef_bin, aft_bin, drw_c, bef_i, aft_i,
                    [bef_cnt_y, bef_cnt_x], [aft_cnt_y, aft_cnt_x], step
                )

        if not bef_covered:
            interpolate_simple_association(
                bef_bin, None, drw_c, bef_i, aft_i,
                [bef_cnt_y, bef_cnt_x], [bef_cnt_y, bef_cnt_x], step
            )

    for j, ac in enumerate(aft_cvg):
        if not ac:
            aft_bin = np.equal(aft_lbl, j+1).astype(np.int)
            aft_cnt_x, aft_cnt_y = np.argwhere(aft_bin == 1).sum(0)/aft_bin.sum()
            interpolate_simple_association(
                None, aft_bin, drw_c, bef_i, aft_i,
                [aft_cnt_y, aft_cnt_x], [aft_cnt_y, aft_cnt_x], step
            )

    """
    # If each only has one candidate, that's easy
    for istr in bef_to_aft:
        if len(bef_to_aft[istr]) == 1 and len(aft_to_bef[str(bef_to_aft[istr][0]["ind"])]) == 1:
            bef_bin = np.equal(bef_lbl, int(istr)).astype(np.int)
            aft_bin = np.equal(aft_lbl, bef_to_aft[istr][0]["ind"]).astype(np.int)
            aft_cnt_x, aft_cnt_y = np.argwhere(aft_bin == 1).sum(0)/aft_bin.sum()
            bef_cnt_x, bef_cnt_y = np.argwhere(bef_bin == 1).sum(0)/bef_bin.sum()
            interpolate_simple_association(
                bef_bin, aft_bin, drw_c, bef_i, aft_i,
                [bef_cnt_y, bef_cnt_x], [aft_cnt_y, aft_cnt_x], step
            )
        else: # More complex decision...
            strict_bta = [x for x in bef_to_aft[istr] if x["ovr_sz"] > 0]
            handled = False
            if len(strict_bta) == 1:
                strict_atb = [x for x in aft_to_bef[str(strict_bta[0]["ind"])] if x["ovr_sz"] > 0]
                if len(strict_atb) == 1:
                    handled = True
                    bef_bin = np.equal(bef_lbl, int(istr)).astype(np.int)
                    aft_bin = np.equal(aft_lbl, strict_bta[0]["ind"]).astype(np.int)
                    aft_cnt_x, aft_cnt_y = np.argwhere(aft_bin == 1).sum(0)/aft_bin.sum()
                    bef_cnt_x, bef_cnt_y = np.argwhere(bef_bin == 1).sum(0)/bef_bin.sum()
                    interpolate_simple_association(
                        bef_bin, aft_bin, drw_c, bef_i, aft_i,
                        [bef_cnt_y, bef_cnt_x], [aft_cnt_y, aft_cnt_x], step
                    )
            if not handled: # Need to do a group merge
                meta = {
                    "istr": istr,
                    "step": step
                }
                # afters = [str(x["ind"]) for x in strict_bta]
                # befores = [str(strict_atb[x]["ind"]) for x in afters]
                dst = Path("/home/helle246/code/repos/sandbox/interpolation/data") / "{}_{}".format(bef_i, aft_i)
                dst.mkdir(exist_ok=True)
                np.save(str(dst / "bef_lbl.npy"), bef_lbl)
                np.save(str(dst / "aft_lbl.npy"), aft_lbl)
                with (dst / "meta.json").open('w') as f:
                    f.write(json.dumps(meta))
                with (dst / "bef_to_aft.json").open('w') as f:
                    f.write(json.dumps(bef_to_aft))
                with (dst / "aft_to_bef.json").open('w') as f:
                    f.write(json.dumps(aft_to_bef))


    """
    return drw_c


def interpolate_drawings(drw_c, step):
    start = 0
    while start < drw_c.shape[0]:
        if np.sum(drw_c[start]) > 0:
            break
        else:
            start += 1

    while start < drw_c.shape[0] + step - 1:
        drw_c = interpolate_step(max(start - step, 0), min(start, drw_c.shape[0] -1), drw_c, step)
        start += step

    return drw_c


def get_blur_kernel_d(affine):
    kerx = signal.gaussian(5, std=1/np.abs(affine[0,2])).reshape(5, 1)
    kerxy = np.outer(kerx, kerx).reshape(1, 5, 5)
    kerz = signal.gaussian(5, std=1/np.abs(affine[2,0])).reshape(5, 1, 1)
    kerxyz = np.outer(kerz, kerxy)
    kerxyz /= np.sum(kerxyz)
    return torch.from_numpy(kerxyz.reshape(1,1,5,5,5)).to("cuda:0")


def get_threshold(region_type):
    # TODO tune this
    if region_type == "ureter":
        return -50
    return -30


def add_renal_hilum(thresholded_d, blr_d, lzn):
    # TODO
    return thresholded_d


def generate_segmentation(region_type, cropped_img, cropped_drw, step=1, affine=None, lzn=None):
    # Interpolate drawings
    cropped_drw = interpolate_drawings(cropped_drw, step)

    # Send tensors to GPU
    img_d = torch.from_numpy(cropped_img).to("cuda:0")
    drw_d = torch.from_numpy(cropped_drw).to("cuda:0")

    # Apply a 3d blur convolution
    blur_kernel_d = get_blur_kernel_d(affine)
    blr_d = torch.nn.functional.conv3d(
        img_d.reshape((1,1)+cropped_img.shape),
        blur_kernel_d, stride=1, padding=2
    ).reshape(cropped_img.shape)

    # Apply threshold
    threshold = get_threshold(region_type)
    thresholded_d = torch.logical_and(
        torch.greater(blr_d, threshold),
        torch.greater(drw_d, 0)
    ).int()
    # thresholded_d = torch.greater(drw_d, 0)

    # If region is kidney, add hilum, redraw, and get new threshold
    if region_type == "kidney":
        thresholded_d = add_renal_hilum(thresholded_d, blr_d, lzn)

    # Bring result back to cpu memory
    return thresholded_d.to("cpu").numpy()


def inflate_seg_to_image_size(cbox, cropped_seg):
    seg_np = np.zeros((cbox["zdim"], cbox["ydim"], cbox["xdim"]), dtype=np.int)
    seg_np[
        cbox["zmin"]:cbox["zmax"] + 1,
        cbox["ymin"]:cbox["ymax"] + 1,
        cbox["xmin"]:cbox["xmax"] + 1,
    ] = cropped_seg
    return seg_np


def delineation_to_seg(region_type, image_path, delineation_path, localization_path=None):
    # Read and parse delination and (maybe) localization from file
    lzn = None
    if region_type == "kidney":
        assert localization_path is not None
        lzn = load_json(localization_path)
    dln = load_json(delineation_path)

    # Read CT scan
    img_nib = nib.load(str(image_path))

    # Crop image to the smallest possible box for memory/computational efficiency
    cbox = get_containing_box(dln, img_nib.shape)
    cropped_img = get_cropped_scan(cbox, img_nib)

    # Generate the drawing made by the annotator
    cropped_drw = generate_cropped_drawing_interior(cbox, dln)

    # Apply heuristics to infer segmentation based on drawing and image
    cropped_seg = generate_segmentation(
        region_type, cropped_img, cropped_drw, cbox["step"], img_nib.affine, lzn
    )

    # Undo cropping to get final segmentation
    seg = inflate_seg_to_image_size(cbox, cropped_seg)

    # Return the seg in nifti format
    return nib.Nifti1Image(seg, img_nib.affine)
