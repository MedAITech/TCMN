import numpy as np
import os
from PIL import Image
import cv2
import math

class EvalMetrix:
    def __init__(self, list_model, list_dataset, exp_id, log_detail=True, save_img=False):
        self.log_detail = log_detail
        self.save_img = save_img
        self.list_model = list_model
        self.list_dataset = list_dataset
        self.exp_id = exp_id
        self.model_index = 0
        self.dataset_index = 0
        self.model = list_model[self.model_index]
        self.dataset = list_dataset[self.dataset_index]
        self.list_masks = []
        self.list_pred = []
        self.obj_video_metrix = []
        self.back_video_metrix = []
        self.obj_dataset_metrix = []
        self.back_dataset_metrix = []
        self.obj_model_metrix = []
        self.back_model_metrix = []

    def put_mask(self, real_mask, pred_mask):
        if real_mask.shape != pred_mask.shape:
            pred_mask = Image.fromarray(pred_mask)
            pred_mask = pred_mask.resize(real_mask.T.shape,Image.BILINEAR)
            pred_mask = np.array(pred_mask)
        self.list_masks.append(real_mask)
        self.list_pred.append(pred_mask)

    def video_save(self, vid_name):
        real_masks = np.array(self.list_masks).astype(np.bool_)
        pred_masks = np.array(self.list_pred).astype(np.bool_)
        this_o = self.new_result(real_masks,pred_masks)  # 计算单个视频的iou和F
        # this_b = self.new_result(np.logical_not(real_masks),np.logical_not(pred_masks))
        # all_pixel = real_masks.shape[0] * real_masks.shape[1] * real_masks.shape[2]
        # IOUo_i = (pred_masks & real_masks).sum()
        # IOUo_u = (pred_masks | real_masks).sum()
        # IOUb_i = all_pixel - IOUo_u
        # IOUb_u = all_pixel - IOUo_i
        # this_o = self.get_result(IOUo_i, IOUo_u, real_masks.sum(), pred_masks.sum())
        # this_b = self.get_result(IOUb_i, IOUb_u, all_pixel - real_masks.sum(), all_pixel - pred_masks.sum())
        self.obj_video_metrix.append(np.array(this_o))
        # self.back_video_metrix.append(np.array(this_b))
        if self.log_detail:
            if not os.path.exists(f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}'):
                os.makedirs(f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}')
            with open(f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}/{self.dataset}.txt', 'a') as f:
                print(
                    '%s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                        self.dataset + ' : ' + vid_name, *this_o, np.mean(this_o[0:2])),
                    file=f)
                # print(
                #     '%s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
                #         self.dataset + ' : ' + vid_name, *this_b),
                #     file=f)
                print('-----------------------------------------------------------------------------------------------',
                      file=f)

        if self.save_img:
            save_path = f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}/{self.dataset}/{vid_name}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(len(pred_masks)):
                img_E = Image.fromarray(pred_masks[i].astype(np.uint8)).convert('P')
                img_E.putpalette([0, 0, 0, 252, 157, 154, 249, 205, 173, 200, 200, 169, 131, 175, 155])
                img_E.save(os.path.join(save_path, (str(i)+'.png').zfill(9)))


        self.list_masks = []
        self.list_pred = []

    def dataset_save(self):
        obj_video_metrix = np.array(self.obj_video_metrix)
        # back_video_metrix = np.array(self.back_video_metrix)
        obj_video_mean = [sum(obj_video_metrix[:, i]) / len(obj_video_metrix) for i in range(len(obj_video_metrix[0]))]
        # back_video_mean = [sum(back_video_metrix[:, i]) / len(back_video_metrix) for i in
        #                    range(len(back_video_metrix[0]))]
        self.obj_dataset_metrix.append(obj_video_mean)
        # self.back_dataset_metrix.append(back_video_mean)
        if self.log_detail:
            with open(f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}/{self.dataset}.txt', 'a') as f:
                print(
                    f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                        self.dataset, *obj_video_mean,np.mean(obj_video_mean[0:2])),
                    file=f)
                # print(
                #     f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
                #         self.dataset, *back_video_mean),
                #     file=f)
        self.obj_video_metrix = []
        self.back_video_metrix = []
        self.dataset_index = (self.dataset_index+1)%len(self.list_dataset)
        self.dataset = self.list_dataset[self.dataset_index]

    def model_save(self):
        with open(f'./{self.exp_id}/{self.model.split("/")[-1].rstrip(".pth")}/All.txt', 'a') as f:
            # for dataset_str, o, b in zip(self.list_dataset, self.obj_dataset_metrix, self.back_dataset_metrix):
            #     print(
            #         f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
            #             dataset_str, *o),
            #         file=f)
            #     print(
            #         f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
            #             dataset_str, *b),
            #         file=f)
            #     print('-----------------------------------------------------------------------------------------------',
            #           file=f)
            for dataset_str, o in zip(self.list_dataset, self.obj_dataset_metrix):
                print(
                    f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                        dataset_str, *o, np.mean(o[0:2])),
                    file=f)
                # print(
                #     f'Dataset %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
                #         dataset_str, *b),
                #     file=f)
                print('-----------------------------------------------------------------------------------------------',
                      file=f)
            obj_dataset_metrix = np.array(self.obj_dataset_metrix)
            # back_dataset_metrix = np.array(self.back_dataset_metrix)
            obj_dataset_mean = [sum(obj_dataset_metrix[:, i]) / len(obj_dataset_metrix) for i in
                                range(len(obj_dataset_metrix[0]))]
            # back_dataset_mean = [sum(back_dataset_metrix[:, i]) / len(back_dataset_metrix) for i in
            #                      range(len(back_dataset_metrix[0]))]
            self.obj_model_metrix.append(obj_dataset_mean)
            # self.back_model_metrix.append(back_dataset_mean)
            print(
                f' Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                    *obj_dataset_mean,np.mean(obj_dataset_mean[0:2])),
                file=f)
            # print(
            #     f' Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (
            #         *back_dataset_mean,),
            #     file=f)
            self.obj_dataset_metrix = []
            self.back_dataset_metrix = []
            self.model_index = (self.model_index+1)%len(self.list_model)
            self.model = self.list_model[self.model_index]

    def all_save(self):
        best = np.zeros((3,3)).tolist()
        best_model = ['']*3
        best_jf = [0,0,0,0]
        best_jf_model = ''
        with open(f'./{self.exp_id}/All.txt', 'a') as f:
            # for model_str, o, b in zip(self.list_model, self.obj_model_metrix, self.back_model_metrix):
            #     if o[1] > best_metrix[1]:
            #         best_model = model_str
            #         best_metrix = o
            #     print(f'Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (model_str, *o),file=f)
            #     print(f'Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (model_str, *b),file=f)
            #     print('-----------------------------------------------------------------------------------------------',
            #           file=f)
            for model_str, o in zip(self.list_model, self.obj_model_metrix):
                for i in range(len(best_model)):
                    if o[i]>best[i][i]:
                        best[i]=o
                        best_model[i]=model_str
                if np.mean(o[0:2])> best_jf[3]:
                    best_jf = o + [np.mean(o[0:2])]
                    best_jf_model = model_str
                print(f'Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (model_str, *o, np.mean(o[0:2])),file=f)
                # print(f'Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f,J&F:%.5f' % (model_str, *b),file=f)
                print('-----------------------------------------------------------------------------------------------',
                      file=f)
            for i in range(len(best_model)):
                print(
                    f'Best_Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                        best_model[i], *best[i], np.mean(best[i][0:2])),
                    file=f)
            print(
                f'Best_Model %s: Jaccard:%.5f, F-measure:%.5f, Dice:%.5f, J&F:%.5f' % (
                    best_jf_model, *best_jf),
                file=f)

    def get_result(self, iou_i, iou_u, real_sum, predict_sum):
        precision = iou_i / predict_sum
        recall = iou_i / real_sum
        f1_score = 2 * precision * recall / (precision + recall)
        dice = 2 * iou_i / (iou_u + iou_i)
        iou = iou_i / iou_u
        return f1_score, iou, precision, recall, dice

    def new_result(self, real_masks, pred_mask):
        j_metrics_res,dice_metrics_res = self.db_eval_iou(real_masks, pred_mask)  # 计算每一帧的iou
        f_metrics_res = self.db_eval_boundary(real_masks, pred_mask)
        JM = np.nanmean(j_metrics_res)  # 视频的所有帧的iou的均值作为视频的iou
        FM = np.nanmean(f_metrics_res)
        DM = np.nanmean(dice_metrics_res)
        return JM,FM,DM

    def db_eval_iou(self, annotation, segmentation, void_pixels=None):
        """ Compute region similarity as the Jaccard Index.
        Arguments:
            annotation   (ndarray): binary annotation   map.
            segmentation (ndarray): binary segmentation map.
            void_pixels  (ndarray): optional mask with void pixels

        Return:
            jaccard (float): region similarity
        """
        assert annotation.shape == segmentation.shape, \
            f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
        annotation = annotation.astype(bool)
        segmentation = segmentation.astype(bool)

        if void_pixels is not None:
            assert annotation.shape == void_pixels.shape, \
                f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
            void_pixels = void_pixels.astype(bool)
        else:
            void_pixels = np.zeros_like(segmentation)

        # Intersection between all sets
        inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
        union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))
        union2 = np.sum(segmentation & np.logical_not(void_pixels),axis=(-2, -1))+np.sum(annotation & np.logical_not(void_pixels),axis=(-2, -1))

        j = inters / union
        dice = 2*inters / union2
        if j.ndim == 0:
            j = 1 if np.isclose(union, 0) else j
        else:
            j[np.isclose(union, 0)] = 1
            dice[np.isclose(union2, 0)] = 1
        return j,dice


    def db_eval_boundary(self, annotation, segmentation, void_pixels=None, bound_th=0.008):
        assert annotation.shape == segmentation.shape
        if void_pixels is not None:
            assert annotation.shape == void_pixels.shape
        if annotation.ndim == 3:
            n_frames = annotation.shape[0]
            f_res = np.zeros(n_frames)
            for frame_id in range(n_frames):
                void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
                f_res[frame_id] = self.f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
        elif annotation.ndim == 2:
            f_res = self.f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
        else:
            raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
        return f_res

    def f_measure(self, foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.

        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.
            void_pixels     (ndarray): optional mask with void pixels

        Returns:
            F (float): boundaries F-measure
        """
        assert np.atleast_3d(foreground_mask).shape[2] == 1
        if void_pixels is not None:
            void_pixels = void_pixels.astype(bool)
        else:
            void_pixels = np.zeros_like(foreground_mask).astype(bool)

        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

        # Get the pixel boundaries of both masks
        fg_boundary = self._seg2bmap(foreground_mask * np.logical_not(void_pixels))
        gt_boundary = self._seg2bmap(gt_mask * np.logical_not(void_pixels))

        from skimage.morphology import disk

        # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
        fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
        # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
        gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

        # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(fg_match) / float(n_fg)
            recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2 * precision * recall / (precision + recall)

        return F

    def _seg2bmap(self, seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
         David Martin <dmartin@eecs.berkeley.edu>
         January 2003
        """

        seg = seg.astype(bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (
                width > w | height > h | abs(ar1 - ar2) > 0.01
        ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap