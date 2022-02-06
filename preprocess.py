import json
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator



def __calculate_bbx (bbx_list):
    x1, y1, w, h = tuple(bbx_list)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x1 + w)
    y2 = int(y1 + h)
    start = (x1, y1)
    end = (x2, y2)
    return start, end

class cocoParser:
    """[summary]
    """
    def __init__(self, cocopath ) -> None:
        
        self.cocopath = cocopath
        #self.imagespath = imagespath
        
        with open(cocopath, 'rb') as f:
            coco_labels = json.load(f)

        self.annotations = coco_labels['annotations']
        self.label_ids = [dct['id'] for dct in coco_labels['annotations'] ]
        self.image_ids = [dct['image_id'] for dct in coco_labels['annotations']]
        self.segs = [dct.get('segmentation') for dct in coco_labels['annotations']]
        self.bbxs = [dct.get('bbox') for dct in coco_labels['annotations']]
        self.names_ids = [(dct['id'], dct['file_name']) for dct in coco_labels['images']]

    
    def postive_data(self):
      
        def __formatsegs (lst):
            segs = [np.array(seg, dtype= np.int32) for seg in lst]
            segs = [seg.reshape(-1,2) for seg in segs]
            return segs

        def __createmask (segs, shape):
            canvas = np.zeros(shape)
            mask = cv.fillPoly(canvas, pts = segs, color = (255, 255, 255))
            return mask

        def __create_border (segs, shape, thickness):
            canvas = np.zeros(shape, dtype= np.float32)
            border = cv.polylines(canvas, segs, True, (255,255,255) , thickness)
            return border

        # retrieve the unique image names and ids with a valid annotation poly in the cocolabels file.
        ids_, names = zip(*self.names_ids)
        valid_names_ids = [(img_id, name) for img_id in set(self.image_ids) for (id_, name) in self.names_ids if img_id == id_]
        valid_ids, valid_names = zip(*valid_names_ids)

        # get a tuple of each valid image id & their annotation polys --> (id, [seg_poly])
        valid_segs = defaultdict(list)
        segs_ids = [(img_id, ann['segmentation']) for img_id in valid_ids for ann in self.annotations if img_id == ann['image_id'] ]

        # collect the multiple segmenation ploys that belong to the same id in a single 2d list per id --> {id: [[ploy_1, poly_2,..,ploy_n]]}
        for img_id, seg in segs_ids:
            valid_segs[img_id].append(seg[0])

        # record in a data frame
        segs = list(valid_segs.values()) # put all the 2d seg lists from the dict in one single list to pass to the data frame
        df_dct = {'img_id': valid_ids, 'name': valid_names, 'seg': segs}
        pos_data = pd.DataFrame(df_dct)

        # format the polygon arrays, create borders and masks
        shape = (256,256)
        pos_data['seg'] = pos_data['seg'].apply(__formatsegs)
        pos_data['mask'] = pos_data['seg'].apply(lambda x: __createmask(x, shape ) )
        pos_data['mask'] = pos_data['seg'].apply(lambda x: __createmask(x, shape ) )
        pos_data['border'] = pos_data['seg'].apply(lambda x: __create_border(x, shape, thickness = 4))
        
        return pos_data


class buildSplits:
    '''
    the class expects the train/val/test split images each in a folder.
    takes in the postive data frame
    returns a dataframe with cols : [index, segmentation coords, mask label array, border label array]
    '''

    def __init__(self, postive_df, split_imgpath) -> None:
        self.postive_df = postive_df
        self.split_imgpath = split_imgpath

    def get_split_df(self, split = 'train'):
        split_df = pd.DataFrame (os.listdir(self.split_imgpath + f'{split}/'), columns = ['name'])
        split_df = pd.merge(split_df, self.postive_df, on= 'name', how= 'left')

        blank_seg = np.zeros_like(self.postive_df.loc[1, 'seg'])
        blank_canvas = np.zeros_like(self.postive_df.loc[1, 'mask'])

        split_df = split_df.replace(np.nan, 0)
        split_df['seg'] = split_df['seg'].map(lambda x: blank_seg if type(x) is int  else x )
        split_df['mask'] = split_df['mask'].map(lambda x: blank_canvas if type(x) is int  else x )
        split_df['border'] = split_df['border'].map(lambda x: blank_canvas if type(x) is int  else x )

        split_df.reset_index(inplace= True, drop= True)

        return split_df

class augmentation:

    # kwargs = {'seed' : 32,
    # 'bright_range': (.4, 1.), 
    # 'hue_range' : 2.0,
    # 'batch_size' : 1,
    # 'rotation': None,
    # 'zoom': None,
    # 'interpolation': None}
    
    def __init__(self, user_dct = None ) -> None:
        
        self.seed = 32
        self.bright_range = (.4, 1.)
        self.hue_range = 2.0
        self.batch_size = 1
        self.rotation = None
        self.zoom = None
        self.interpolation = None
        
        if user_dct:
            for k, v in user_dct.items():
                if k in self.__dict__:
                    setattr(self, k, v)
                else:
                    raise KeyError(k)
        aug_args = dict(
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range = self.bright_range,
                channel_shift_range = self.hue_range,
                rescale=1./255
                             )

        self.gen = ImageDataGenerator(**aug_args)
        self.testgen = ImageDataGenerator(rescale= 1./255)
        keys =  ['seed', 'bright_range', 'hue_range' ,'batch_size', 'rotation' ,'zoom', 'interpolation']
        self.aug_dct = {key: self.__dict__.get(key) for key in keys}
        

    @staticmethod
    def __data_gen (x_gen, mask_gen, border_gen):
        while True:
            image = next(x_gen)
            ymask = next(mask_gen)
            yborder = next(border_gen)
            yield image, [ymask, yborder]

    @staticmethod
    def __get_targetarrays(split_df, target_col = 'mask'):
        size = len(split_df)
        labels = np.ndarray(shape = (size, 256, 256, 1), dtype= np.float32)
        
        for indx in range(size):
            y = split_df.loc[indx, target_col]
            y  = y.reshape((256,256,1))
            labels[indx] = y 
        return labels

    
    def get_splitgen (self, split_df, imgpath, test = False):
        gen = self.testgen if test else self.gen

        masks_arrs = self.__get_targetarrays(split_df)
        borders_arrs = self.__get_targetarrays(split_df, target_col = 'border')
        
        x_flow = gen.flow_from_dataframe(split_df, x_col = 'name', class_mode= None, validate_filenames= True,
         directory= imgpath, batch_size=self.batch_size, seed= self.seed)

        y_maskflow  = gen.flow(masks_arrs, batch_size=self.batch_size, seed= self.seed)
        y_borderflow  = gen.flow(borders_arrs, batch_size=self.batch_size, seed= self.seed)
        split_gen = self.__data_gen(x_flow, y_maskflow, y_borderflow)

        return split_gen





def label_overlay(ximg, mask, border, pred = False):
    
    pixel_clip = np.vectorize(lambda pixel: 0 if pixel < 0.5 else 1)

    mask_clr, brdr_clr = (255, 0, 0) , (0, 255, 0)
    if pred: 
        mask_clr, brdr_clr = (0, 0, 255), (255, 255, 255) 
        mask, border  = (pixel_clip(label).squeeze() for label in (mask, border))

    image = np.copy(ximg)
    image /=  np.max(image)
    image *= 255
    image = image.astype(dtype = np.uint8).squeeze()

    mask *= 255
    border *= 255
    mask = mask.astype(dtype = np.uint8).squeeze()
    border = border.astype(dtype = np.uint8).squeeze()
    
    blue_canvas = np.full(image.shape, mask_clr, image.dtype)
    white_canvas = np.full(image.shape, brdr_clr, image.dtype)

    blueMask = cv.bitwise_and(blue_canvas, blue_canvas, mask=mask)
    whiteborder = cv.bitwise_and(white_canvas, white_canvas, mask=border)
    out = cv.addWeighted(blueMask, .5, image, 1, 0, image)
    out = cv.addWeighted(whiteborder, .5, out, 1, 0, out)
    return out



def single_overlay(ximg, mask, border):
    
    image = np.copy(ximg)
    image /=  np.max(image)
    image  *= 255
    image = image.astype(dtype = np.uint8).squeeze()
    mask *= 255
    mask = mask.astype(dtype = np.uint8).squeeze()
    border *= 255
    border = border.astype(dtype = np.uint8).squeeze()

    
    red_canvas = np.full(image.shape, (255, 0, 0), image.dtype)
    green_canvas = np.full(image.shape, (0, 255, 0), image.dtype)

    redMask = cv.bitwise_and(red_canvas, red_canvas, mask=mask)
    greenborder = cv.bitwise_and(green_canvas, green_canvas, mask=border)
    out = cv.addWeighted(redMask, .5, image, 1, 0, image)
    out = cv.addWeighted(greenborder, .5, out, 1, 0, image)
    return out


def cocodraw(self, size, index = 0,*, image_num = None):

    
    if image_num:
        image_name = str(image_num)
        image_id = [img_id for (img_id, name) in self.names_ids if image_name in name][0]
        annotation = [dct for dct in self.annotations if dct['image_id'] == image_id]
    else:
        image_id = [dct['image_id'] for dct in self.annotations if dct['id'] == index][0]
        annotation = [dct for dct in self.annotations if dct['image_id'] == image_id]

    label_id = annotation[0]['id']
    img_id = annotation[0]['image_id']
    segs = [dct['segmentation'] for dct in annotation]
    bbxes = [dct['bbox'] for dct in annotation]
    img_name = [name for id, name in self.names_ids if id == img_id][0]
    img_path = Path(self.imagespath + img_name)

    segs = [np.array(seg, dtype= np.int32) for seg in segs]
    segs = [seg.reshape(-1,2) for seg in segs]
    
    bbxes = [__calculate_bbx(box) for box in bbxes]

    try:
        img_path = img_path.resolve(strict=True)
    except FileNotFoundError as err:
        print(err)
    else:
        img = cv.imread(img_path.as_posix())
        img_seg = np.copy(img)
        img_bbx = np.copy(img)
        img_seg = cv.polylines(img_seg, segs, True,(36,255,12) , 2)
        for box in bbxes:
            start, end = box
            cv.rectangle(img_bbx, start, end,(36,255,12) , 2)


        plt.figure(figsize= size)
        plt.subplot(121)
        plt.imshow(img_seg)
        plt.title (f"Id: {label_id} | Img_id: {img_id} | {img_name}")

        plt.subplot(122 )
        plt.imshow(img_bbx)
        plt.show()





