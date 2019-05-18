from __future__ import division
import time
from util import *
from darknet import Darknet
import random
import pickle as pkl

def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return label

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    confidence = 0.25
    nms_thresh = 0.4
    reso = "160"

    confidence = float(confidence)
    nms_thesh = float(nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    cap = cv2.VideoCapture(0)

    frames = 0
    start = time.time()

    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1, 2)
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)


            if type(output) == int:
                frames += 1
                cv2.putText(orig_im, str("{:5.2f}".format(frames / (time.time() - start))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim)) / inp_dim
            
            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            print(list(map(lambda x: write(x, orig_im), output)))

            cv2.putText(orig_im, str("{:5.2f}".format(frames / (time.time() - start))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1