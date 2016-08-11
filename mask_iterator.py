from neon.data import DataLoader, ImageLoader
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

#shape = dict(channel_count=3, height=32, width=32)
#train_params = ImageParams(center=False, aspect_ratio=110, **shape)
#test_params = ImageParams(**shape)
#common = dict(target_size=1, nclasses=10)
#
#train = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
#                           shuffle=True, subset_percent=args.subset_pct, **common)
#test = DataLoader(set_name='val', repo_dir=test_dir, media_params=test_params, **common)
#tune_set = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
#                              subset_percent=20, **common)

import numpy as np

class MaskIterator(ImageLoader):
    """
    Generates an image and some text.  The text is overlaid
    The target is the text mask
    """


    def __init__(self, min_len, max_len, reps, *args, **kwargs):
        super(MaskIterator, self).__init__(*args, **kwargs)
        np.random.seed(1)
        self.min_len = min_len
        self.max_len = max_len
        self.reps = reps
        self.out_full = None
        self.t_int32 = None
        self.tview = None
        self.nclasses = 2  # text or image
        self.text_mask = self.be.iobuf(int(np.prod(self.shape)))
        self.text = self.be.iobuf(int(np.prod(self.shape)))
        self.text_mask_host = None #np.zeros(self.text_mask.shape)
        self.text_host = None #np.zeros(self.text_mask.shape)

        self.means = np.array([104, 119, 126])  # RGB means for I1K

    def gen_text_masks(self):
        if self.text_mask_host is None:
            self.text_mask_host = np.zeros(self.text_mask.shape)
            self.text_host = np.zeros(self.text_mask.shape)
        else:
            self.text_mask_host[:] = 0
            self.text_host[:] = 0
        im_shape = list(self.shape[1:])
        #lens = np.random.randint(self.min_len, self.max_len+1, self.be.bsz)
        for bind in range(self.be.bsz):
            #len_ = lens[bind]
            #                        W,   H       white bgnd
            # for b/W 
            #img = Image.new('RGB', im_shape[::-1], (255, 255, 255))
            img = Image.new('RGB', im_shape[::-1], (0, 0, 0))
            for rind in range(self.reps):
                len_ = np.random.randint(self.min_len, self.max_len+1, 1)
                chars = [chr(xx_) for xx_ in np.random.randint(97, 97+26-1, len_)] 
                chars[0] = chars[0].upper()
                txt = ''.join(chars)

                fsize = int(np.random.randint(10, 30, 1)[0])
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", fsize)

                d = ImageDraw.Draw(img)
                # text_w is along the x dim when plotted
                # text_h is alonf the y dim when plotted
                text_w, text_h = d.textsize(txt, font=font)

                mx_coord = np.array([im_shape[0]-text_h, im_shape[1]-text_w])
                mx_coord = [np.max([0, xx_]) for xx_ in mx_coord]
                mx_coord = np.array(mx_coord)

                coord = [np.random.uniform(low=0.0, high=mx_coord[0]),
                         np.random.uniform(low=0.0, high=mx_coord[1])]
                # no clr: clr = (0,0,0)
                #d.text((coord[1], coord[0]), txt, fill=(0,0,0))
                clr = tuple(np.random.randint(1, 256, 3))
                d.text((coord[1], coord[0]), txt, fill=clr, font=font)

            # b/w
            #txt_mask = np.array(img)/255
            img_arr = np.array(img)
            #img_arr = img_arr[:,:,::-1]  # go to RGB  just skip it color is random
            txt_mask = img_arr.copy()
            txt_mask = txt_mask > 0
            txt_mask = 1 - txt_mask.astype(np.uint8)
            img_arr = img_arr.astype(np.float32) - self.means

            #if bind == 1:
            #    self.temp = txt_mask
            #    self.img = img
            self.text_mask_host[:,bind] = txt_mask.astype(np.float32).transpose((2,0,1)).flatten() 
            self.text_host[:,bind] = img_arr.transpose((2,0,1)).flatten() 
        # copy to dev
        self.text_mask[:] = self.text_mask_host
        self.text[:] = self.text_host


    def next(self, start):
        # first get the images from the dataloader for this minibatch
        x, t  = super(MaskIterator, self).next(start)

        self.gen_text_masks()

        if self.out_full is None:
            # generate the extra buffers needed to hold the expanded
            # one-hot target output
            self.out_full = self.be.iobuf(self.nclasses*int(np.prod(self.shape[1:])),
                                          dtype=np.int32)
            self.outview = self.out_full.reshape((self.nclasses, -1))

            # need an int32 copy of the target classes for one-hot to work correctly
            self.t_int32 = self.be.iobuf(int(np.prod(self.shape[1:])), dtype=np.int32)
            self.tview = self.t_int32.reshape((1, -1))
        # copy to int32
        tmp = self.text_mask.reshape((3, -1))[0].reshape((int(np.prod(self.shape[1:])), -1))
        self.t_int32[:] = 1 - tmp
        # use backend one-hot to expand output to nclasses x H x W
        self.be.onehot(self.tview, axis=0, out=self.outview)

        x[:] = x*self.text_mask + self.text
        return (x, self.out_full)
