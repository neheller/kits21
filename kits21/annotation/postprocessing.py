"""Code for turning user delineations into dense segmentations."""
import json

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from numpy.core.fromnumeric import cumsum
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

    abs_zmin = 0
    abs_zmax = shape[0] - 1
    return {
        "xmin": max(0, int(np.floor(mins[0] - max_sz))),
        "xmax": min(shape[2] - 1, int(np.ceil(maxs[0] + max_sz))),
        "ymin": max(0, int(np.floor(mins[1] - max_sz))),
        "ymax": min(shape[1] - 1, int(np.ceil(maxs[1] + max_sz))),
        "zmin": max(abs_zmin, min(afrms) - min_step),
        "zmax": min(abs_zmax, max(afrms) + min_step),
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
                ImageDraw.floodfill(im, (0,0), 128, thresh=63.5)
                rszd = im.resize((ret.shape[2], ret.shape[1]), Image.BILINEAR)
                ret[i,:,:] = np.less(np.array(rszd), 63.9).astype(np.int)

    return ret


def get_contour(bin_seg):
    if bin_seg is None:
        return None
    contours, hierarchy = cv2.findContours(bin_seg.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def distance(p1, p2):
    return (p1[0][0] - p2[0][0])*(p1[0][0] - p2[0][0]) + (p1[0][1] - p2[0][1])*(p1[0][1] - p2[0][1])


def find_nearest_neighbors_slow_v2(lg_cntr, sm_cntr):
    matches = np.zeros_like(lg_cntr)
    step = sm_cntr.shape[0]/lg_cntr.shape[0]
    mini = None
    mind = np.inf
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


def get_group(istr, bef_to_aft, aft_to_bef):
    bef_grp = set([istr])
    aft_grp = set([])
    bef_ln = len(bef_grp)
    aft_ln = len(aft_grp)
    while True:
        for ai in aft_grp:
            for atb in aft_to_bef[ai]:
                if atb["ovr_sz"] > 0:
                    bef_grp.add(str(atb["ind"]))
        for bi in bef_grp:
            for bta in bef_to_aft[bi]:
                if bta["ovr_sz"] > 0:
                    aft_grp.add(str(bta["ind"]))
        if len(bef_grp) != bef_ln or len(aft_grp) != aft_ln:
            bef_ln = len(bef_grp)
            aft_ln = len(aft_grp)
        else:
            break
    return list(bef_grp), list(aft_grp)


def splice_contour(spliced, stretches, cntr, cur_sz, ctr_ind):
    # Get nearest pair
    mini = None
    minj = None
    mind = np.inf
    for i in range(cur_sz):
        for j in range(cntr.shape[0]):
            dst = distance(spliced[i], cntr[j])
            if dst < mind:
                mini = i
                minj = j
                mind = dst
                                
    ret_sp = spliced.copy()
    ret_sp[mini+1:mini+cntr.shape[0]+1] = cntr
    ret_sp[mini+cntr.shape[0]+1:cur_sz+cntr.shape[0]] = spliced[mini+1:cur_sz]

    ret_st = stretches.copy()
    ret_st[mini+1:mini+cntr.shape[0]+1] = ctr_ind*np.ones((cntr.shape[0], 1))
    ret_st[mini+cntr.shape[0]+1:cur_sz+cntr.shape[0]] = stretches[mini+1:cur_sz]
    
    return ret_sp, ret_st
    

def splice_contours(cntrs):
    lengths = [cr.shape[0] for cr in cntrs]
    stretches = -1*np.ones(
        (sum(lengths),1),
        dtype=np.int32
    )
    
    spliced = np.zeros(
        (sum(lengths),) + cntrs[0].shape[1:],
        dtype=cntrs[0].dtype
    )
    spliced[0:cntrs[0].shape[0]] = cntrs[0].copy()
    stretches[0:cntrs[0].shape[0]] = np.zeros((cntrs[0].shape[0], 1))
    for i in range(1, len(cntrs)):
        spliced, stretches = splice_contour(spliced, stretches, cntrs[i], sum(lengths[:i]), i)

    return spliced, stretches


def slice_matches(matches, splice_inds):
    ret = []
    for i in range(np.max(splice_inds)+1):
        ret += [matches[splice_inds == i,:].reshape((-1,1,2))]
    
    return ret 


def interpolate_merge_association(bef_grp, aft_grp, bef_lbl, aft_lbl, drw_c, bef_i, aft_i, step):
    # Get composites for each
    tot_bef_bin = np.zeros_like(bef_lbl)
    for lbl in bef_grp:
        tot_bef_bin = np.logical_or(
            tot_bef_bin,
            np.equal(bef_lbl, int(lbl))
        )
    tot_aft_bin = np.zeros_like(aft_lbl)
    for lbl in aft_grp:
        tot_aft_bin = np.logical_or(
            tot_aft_bin,
            np.equal(aft_lbl, int(lbl))
        )
        
    # Get individual values
    bef_bins = [
        np.equal(bef_lbl, int(x))
        for x in bef_grp
    ]
    aft_bins = [
        np.equal(aft_lbl, int(x))
        for x in aft_grp
    ]
    bef_cntrs = [
        get_contour(bef_bin)
        for bef_bin in bef_bins
    ]
    aft_cntrs = [
        get_contour(aft_bin)
        for aft_bin in aft_bins
    ]
    if len(bef_grp) > len(aft_grp):
        nonref_cntrs = bef_cntrs
        spliced_nonref, splice_inds = splice_contours(bef_cntrs)
        ref_cntrs = aft_cntrs
        start = aft_i
        inc = -1
    else:
        nonref_cntrs = aft_cntrs
        spliced_nonref, splice_inds = splice_contours(aft_cntrs)
        ref_cntrs = bef_cntrs
        start = bef_i
        inc = 1

    for ref_cntr in ref_cntrs:
        matches = find_nearest_neighbors_slow_v2(ref_cntr, spliced_nonref)
        rev_matches = find_nearest_neighbors_slow_v2(spliced_nonref, ref_cntr)
        sliced_matches = slice_matches(rev_matches, splice_inds)
        for i in range(1, int(np.ceil((aft_i - bef_i)/2))):
            draw_filled_contour(
                start + i*inc, bef_i, aft_i,
                drw_c, tot_bef_bin, tot_aft_bin,
                i/step*matches + (step - i)/step*ref_cntr
            )
        for nonref_frag, ref_frag in zip(nonref_cntrs, sliced_matches):
            for i in range(int(np.ceil((aft_i - bef_i)/2)), aft_i - bef_i):
                draw_filled_contour(
                    start + i*inc, bef_i, aft_i,
                    drw_c, tot_bef_bin, tot_aft_bin,
                    i/step*nonref_frag + (step - i)/step*ref_frag
                )


def interpolate_simple_association(bef_bin, aft_bin, drw_c, bef_i, aft_i, bef_cnt, aft_cnt, step):
    # cnt <- center
    # cntr <- contour
    bef_cntr = get_contour(bef_bin)
    aft_cntr = get_contour(aft_bin)
    if bef_cntr is None:
        start = bef_i
        inc = 1
        ref = bef_cntr
        bef_cntr = np.array([
            [bef_cnt]
        ])
        bef_bin = np.zeros_like(aft_bin)
    elif aft_cntr is None:
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
        matches = find_nearest_neighbors_slow_v2(bef_cntr, aft_cntr)
    else:
        start = aft_i
        inc = -1
        ref = aft_cntr
        matches = find_nearest_neighbors_slow_v2(aft_cntr, bef_cntr)

    for i in range(1, aft_i - bef_i):
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
            strict_atb = []
            for k in range(len(strict_bta)):
                strict_atb += [
                    x for x in aft_to_bef[str(strict_bta[k]["ind"])] 
                    if x["ovr_sz"] > 0
                ]
            handled = False
            if len(strict_bta) == 1:
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
                bef_grp, aft_grp = get_group(istr, bef_to_aft, aft_to_bef)
                interpolate_merge_association(
                    bef_grp, aft_grp, bef_lbl, aft_lbl, drw_c, bef_i, aft_i, step
                )

    return drw_c


def interpolate_drawings(drw_c, step, arb_bdry=False):
    # Get inclusive start and end frames
    start = 0
    while start < drw_c.shape[0]:
        if np.sum(drw_c[start]) > 0:
            break
        else:
            start += 1
    end = drw_c.shape[0] - 1
    while end > start:
        if np.sum(drw_c[end]) > 0:
            break
        else:
            end -= 1


    if arb_bdry:
        start += step
        end -= step

    while start < end + step + 1:
        drw_c = interpolate_step(max(start - step, 0), min(start, drw_c.shape[0] - 1), drw_c, step)
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
    # This seems to work -- no need to adjust based on region now that ureter is gone
    return -30


def find_hilum_in_slice(thresh, side):
    # TODO use custom if available
    thresh = thresh.astype(np.uint8)
    (
        nb_components, output, stats, centroids
    ) = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    sizes = stats[:, -1]

    max_label = 0
    max_size = 0
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    thresh[output != max_label] = 0
    centroid = np.array(tuple(centroids[max_label]))

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    primary_contour = contours[0]
    hull = cv2.convexHull(primary_contour, returnPoints=False)
    defects = cv2.convexityDefects(primary_contour, hull)

    # Choose from defects
    distances = []
    scores = []
    criteria = []
    depths = []

    if defects is None:
        return None

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = np.array(tuple(primary_contour[s][0]))
        end = np.array(tuple(primary_contour[e][0]))
        furthest = np.array(tuple(primary_contour[f][0]))
        defect_center = (start + end)/2
        depth = np.linalg.norm(furthest - defect_center)
        centroid_offset = centroid - defect_center
        distance = np.linalg.norm(start - end, ord=2)
        # print(centroid, defect_center, centroid_offset, distance)
        if side == "left":
            score = 1*centroid_offset[0] + centroid_offset[1]
        elif side == "right":
            score = -1*centroid_offset[0] + centroid_offset[1]
        distance = np.linalg.norm(start - end, ord=2)
        scores = scores + [score]
        distances = distances + [distance]
        depths = depths + [depth]
        criteria = criteria + [int(score>0)*(distance+3*depth)]

    if np.sum(criteria) > 1e-2:
        winner = np.argmax(criteria)
        s, e, f, d = defects[winner, 0]
        start = tuple(primary_contour[s][0])
        end = tuple(primary_contour[e][0])
        hlm = [start, end]
    else:
        hlm = None

    return hlm


def apply_hilum_to_slice(thresholded_c, blur_c, threshold, ind, hlm):
    if hlm is None:
        return

    cv2.line(thresholded_c[ind], hlm[0], hlm[1], 1, 2)
    abuse_slc = thresholded_c[ind].copy()
    mask = np.zeros((thresholded_c.shape[1]+2, thresholded_c.shape[2]+2), np.uint8)
    cv2.floodFill(abuse_slc, mask, (0,0), 1)
    thresholded_c[ind] = np.logical_and(
        (np.equal(abuse_slc, 0) | thresholded_c[ind]).astype(thresholded_c[ind].dtype),
        np.greater(blur_c[ind], threshold)
    )


# TODO allow for custom hilums to be specified in dln
# Polygons will be allowed for logged-in users
def add_renal_hilum(thresholded_c, blr_c, threshold, lzn, side, cbox, custom_hilums):
    first_hilum_slice = None
    last_hilum_slice = None
    for ann in lzn["annotations"]:
        if ann["spatial_type"] == "whole-image" and not ann["deprecated"]:
            bound = None
            for cp in ann["classification_payloads"]:
                if cp["confidence"] > 0.5:
                    if cp["class_id"] == 7:
                        bound = "sup"
                    elif cp["class_id"] == 8:
                        bound = "inf"
            if bound is None:
                continue
            frame = int(ann["frame"])
            if bound == "sup":
                if first_hilum_slice is None or frame < first_hilum_slice:
                    first_hilum_slice = frame - cbox["zmin"]
            elif bound == "inf":
                if last_hilum_slice is None or frame > last_hilum_slice:
                    last_hilum_slice = frame - cbox["zmin"]

    for ind in range(thresholded_c.shape[0]):
        if "slice_{}".format(ind) in custom_hilums:
            for hlm in custom_hilums["slice_{}".format(ind)]:
                apply_hilum_to_slice(thresholded_c, blr_c, threshold, ind, hlm)
        elif (
            (
                first_hilum_slice is not None and ind >= first_hilum_slice
            ) and (
                last_hilum_slice is not None and ind <= last_hilum_slice
            )
        ):
            # TODO send dln here and use custom hilum if possible
            hlm = find_hilum_in_slice(thresholded_c[ind].copy(), side)
            apply_hilum_to_slice(thresholded_c, blr_c, threshold, ind, hlm)
    else:
        if first_hilum_slice is None:
            print("First hilum slice could not be determined")
        if last_hilum_slice is None:
            print("Last hilum slice could not be determined")

    return thresholded_c


def get_side(cbox):
    if cbox["xmin"] + cbox["xmax"] > cbox["xdim"]:
        return "left"
    return "right"


def generate_segmentation(region_type, cropped_img, cropped_drw, step=1, affine=None, lzn=None, cbox=None, custom_hilums={}):
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
    thresholded_c = thresholded_d.to("cpu").numpy()
    blr_c = blr_d.to("cpu").numpy()
    if region_type == "kidney":
        side = get_side(cbox)
        thresholded_c = add_renal_hilum(thresholded_c, blr_c, threshold, lzn, side, cbox, custom_hilums)

    # Bring result back to cpu memory
    return thresholded_c


def inflate_seg_to_image_size(cbox, cropped_seg):
    seg_np = np.zeros((cbox["zdim"], cbox["ydim"], cbox["xdim"]), dtype=np.int)
    seg_np[
        cbox["zmin"]:cbox["zmax"] + 1,
        cbox["ymin"]:cbox["ymax"] + 1,
        cbox["xmin"]:cbox["xmax"] + 1,
    ] = cropped_seg
    return seg_np


def get_custom_hilums(meta, cbox):
    ret = {}
    if "custom_hilums" not in meta:
        return ret

    for ch in meta["custom_hilums"]:
        if ch["slice_index"] < cbox["zmin"] or ch["slice_index"] > cbox["zmax"]:
            continue

        dct_key = "slice_{}".format(ch["slice_index"] - cbox["zmin"])
        if dct_key not in ret:
            ret[dct_key] = []

        for hlm in ch["hilums"]:
            ret[dct_key] += [
                [
                    (
                        hlm[0][0] - cbox["xmin"],
                        hlm[0][1] - cbox["ymin"]
                    ),
                    (
                        hlm[1][0] - cbox["xmin"],
                        hlm[1][1] - cbox["ymin"]
                    )
                ]
            ]

    return ret


def delineation_to_seg(region_type, image_path, delineation_path, meta, localization_path=None):
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

    # Get any custom hilums within the containing box
    custom_hilums = get_custom_hilums(meta, cbox)

    # Apply heuristics to infer segmentation based on drawing and image
    cropped_seg = generate_segmentation(
        region_type, cropped_img, cropped_drw, cbox["step"], img_nib.affine, lzn, cbox, custom_hilums
    )

    # Undo cropping to get final segmentation
    seg = inflate_seg_to_image_size(cbox, cropped_seg)

    # Return the seg in nifti format
    return nib.Nifti1Image(seg.astype(np.uint8), img_nib.affine)
