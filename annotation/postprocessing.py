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
                        width=int(round(stroke["line_size"]*10)),
                        joint="curve"
                    )
                    srt = stroke["spatial_payload"][0]
                    draw.ellipse(
                        [
                            (
                                int(round((srt[0] - cbox["xmin"] - stroke["line_size"]/2)*10)),
                                int(round((srt[1] - cbox["ymin"] - stroke["line_size"]/2)*10))
                            ),
                            (
                                int(round((srt[0] - cbox["xmin"] + stroke["line_size"]/2)*10)),
                                int(round((srt[1] - cbox["ymin"] + stroke["line_size"]/2)*10))
                            )
                        ],
                        fill=128
                    )
                    end = stroke["spatial_payload"][-1]
                    draw.ellipse(
                        [
                            (
                                int(round((end[0] - cbox["xmin"] - stroke["line_size"]/2)*10)),
                                int(round((end[1] - cbox["ymin"] - stroke["line_size"]/2)*10))
                            ),
                            (
                                int(round((end[0] - cbox["xmin"] + stroke["line_size"]/2)*10)),
                                int(round((end[1] - cbox["ymin"] + stroke["line_size"]/2)*10))
                            )
                        ],
                        fill=128
                    )
            if drew:
                rszd = im.resize((ret.shape[2], ret.shape[1]), Image.BILINEAR)
                ImageDraw.floodfill(rszd, (0,0), 128, thresh=63.5)
                ret[i,:,:] = np.less(np.array(rszd), 63.9).astype(np.int)

    return ret


def interpolate_association(bef_bin, aft_bin, drw_c, bef_i, aft_i):
    # TODO
    pass


def interpolate_step(bef_i, aft_i, drw_c):
    # Label connected components in each
    bef_lbl = measure.label(drw_c[bef_i, :, :], background=0)
    aft_lbl = measure.label(drw_c[aft_i, :, :], background=0)

    # Associate connected components based on proximity and overlap
    num_bef = np.max(bef_lbl)
    num_aft = np.max(aft_lbl)

    aft_cvg = [False for _ in range(num_aft)]

    # Iterate over all pairs of blobs
    for i in range(1, num_bef+1):
        bef_bin = np.equal(bef_lbl, i).astype(np.int)
        bef_cnt_y, bef_cnt_x = np.argwhere(bef_bin == 1).sum(0)/bef_bin.sum()
        bef_covered = False
        for j in range(1, num_aft+1):
            aft_bin = np.equal(aft_lbl, j).astype(np.int)

            # Get size of overlap
            ovr_sz = np.multiply(bef_bin, aft_bin).sum()

            # Get metrics describing blob proximity
            aft_cnt_y, aft_cnt_x = np.argwhere(aft_bin == 1).sum(0)/aft_bin.sum()
            cnt_dsp = [aft_cnt_y - bef_cnt_y, aft_cnt_x - bef_cnt_x]
            cnt_dst_sq = cnt_dsp[0]**2 + cnt_dsp[1]**2

            if ovr_sz > 0 or cnt_dst_sq < 15**2:
                bef_covered = True
                aft_cvg[j-1] = True
                interpolate_association(bef_bin, aft_bin, drw_c, bef_i, aft_i)

        if not bef_covered:
            interpolate_association(bef_bin, None, drw_c, bef_i, aft_i)

    for j, ac in enumerate(aft_cvg):
        if not ac:
            aft_bin = np.equal(aft_lbl, j+1).astype(np.int)
            interpolate_association(None, aft_bin, drw_c, bef_i, aft_i)

    return drw_c


def interpolate_drawings(drw_c, step):
    start = 0
    while start < drw_c.shape[0]:
        if np.sum(drw_c[start]) > 0:
            break
        else:
            start += 1

    while start < drw_c.shape[0] + step - 1:
        drw_c = interpolate_step(max(start - step, 0), min(start, drw_c.shape[0] -1), drw_c)
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
