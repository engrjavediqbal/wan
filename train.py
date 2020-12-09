import argparse
import numpy as np
import os
import os.path as osp
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from unet.model import unet_Model
import timeit
import util

start = timeit.default_timer()

BATCH_SIZE = 8
DATA_DIRECTORY = './datasets/rwanda'
DATA_LIST_PATH_TRAIN = './datasets/dataLists/rwanda/train.txt'
DATA_LIST_PATH_VAL = './datasets/dataLists/rwanda/val.txt'
INPUT_SIZE = '256,256'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
SNAPSHOT_DIR = './snapshots/rwanda/src_train/'
WEIGHT_DECAY = 1e-6
MODEL = 'Unet'
LOG_FILE = 'log'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list-train", type=str, default=DATA_LIST_PATH_TRAIN,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-list-val", type=str, default=DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="The base network.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training steps.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    return parser.parse_args()

args = get_arguments()


def load_images(train_indices, batch_size, itr, lines, image_size, data_dir):
    # Create empty arrays to contain batch of features and labels#
    batch_images = []
    batch_labels = []
    startInd = itr*batch_size
    endInd = startInd + batch_size
    
    for i in range(startInd, endInd):
        line = lines[train_indices[i]].split(' ')

        img = cv2.imread(osp.join(data_dir, line[0]))
        lbl = cv2.imread(osp.join(data_dir, line[1]))
        img = img[...,::-1]

        lebelImage = lbl[:,:,0]
        binLabel = np.zeros((lebelImage.shape), dtype=int)
        binLabel[lebelImage==255] = 1
        
        binLabel = np.reshape(binLabel, (image_size, image_size,1))
        batch_labels.append(binLabel)
        batch_images.append(img)

    
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels, dtype=np.int)
    return batch_images, batch_labels

def executeModelonTest(model, valDataset, data_dir):
    image_size = 256

    linesValid = [line.rstrip('\n') for line in open(valDataset)]

    indicesVal = np.arange(len(linesValid))
    np.random.shuffle(indicesVal) 
    allProb = []
    precAll = []
    recAll = []
    ret_states = []
    ep = 1e-12

    confMatAll = np.zeros(shape=(2, 2),dtype=np.ulonglong)
    
    for j in range(len(linesValid)):

        test_image, y_train= load_images(indicesVal, 1, j, linesValid, image_size, data_dir)
        output = model.predict(test_image)
        output = np.reshape(output, (image_size,image_size))
        outPt = 1*(output>=0.5)
        same = 1*(outPt == y_train[0,:,:,0])
        rec = ((outPt * y_train[0,:,:,0]).sum()+1)/((y_train[0,:,:,0]).sum()+1)
        pre = ((outPt * y_train[0,:,:,0]).sum()+1)/(outPt.sum()+1)
        iou = ((outPt * y_train[0,:,:,0]).sum()+1)/((1*((outPt+y_train[0,:,:,0])>=1.0)).sum()+1)

        allProb.append(same.mean())
        confMat = confusion_matrix(y_train[0,:,:,0].flatten(), outPt.flatten(), labels=range(2))
        confMatAll = confMatAll + confMat
        
    sumH = np.sum(confMatAll, axis = 0)
    sumV = np.sum(confMatAll, axis = 1)
    
    for idd in range(2):
        union = sumH[idd] + sumV[idd] - confMatAll[idd, idd]
        ret_states.append(((ep+confMatAll[idd, idd]))/(union))
        # if (union)>0:
        #     print('IoU class :', (confMatAll[idd, idd])/(union))
            
    prec = confMatAll[1, 1]/sumV[1]
    rec = confMatAll[1, 1]/sumH[1]
    F1 = 2*(prec*rec)/(prec+rec)
    # print('Recall :', rec)
    # print('Prec :', prec)
    # print('F1-Score :', F1)
    ret_states.append(prec)
    ret_states.append(rec)
    ret_states.append(F1)

    return ret_states


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    logger = util.set_logger(args.snapshot_dir, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Create network.

    model = unet_Model(input_size, args.learning_rate, args.weight_decay)

    lines = [line.rstrip('\n') for line in open(args.data_list_train)]
    lines1 = [line.rstrip('\n') for line in open(args.data_list_val)]

    indicesTrain = np.arange(len(lines))
    indicesVal = np.arange(len(lines1))


    print(indicesTrain.shape)
    print(indicesVal.shape)


    mIoUPrev = 0
    mIoUPrev2 = 0

    for epoch in range(args.num_epochs):
        
        np.random.shuffle(indicesTrain)   #shuffle training
        np.random.shuffle(indicesVal)   #shuffle validation

        iterations_t = np.int(len(indicesTrain)/args.batch_size)
        iterations_v = np.int(len(indicesVal)/args.batch_size)

        ## training
        t_loss = []
        t_acc = []
        v_loss = []
        v_acc = []
        for i in range(iterations_t):
            X_train, y_train = load_images(indicesTrain, args.batch_size, i, lines, input_size[0], args.data_dir)
    #         print(X_train.shape)
            history = model.train_on_batch(X_train, y_train)
            t_loss.append(history[0])
            t_acc.append(history[1])
        t_loss = np.mean(np.array(t_loss, dtype=np.float32))
        t_acc = np.mean(np.array(t_acc, dtype=np.float32))  
        print("Epoch: {}  Training Loss: {}  Training Acc: {}".format(epoch, t_loss, t_acc) )
        ## validation
        for j in range(iterations_v):
            X_valid, y_valid = load_images(indicesVal, 1, j, lines1, input_size[0], args.data_dir)
            [a, b] = model.evaluate(X_valid, y_valid, verbose=0)
            
            v_loss.append(a)
            v_acc.append(b)
        v_loss = np.mean(np.array(v_loss, dtype=np.float32))
        v_acc = np.mean(np.array(v_acc, dtype=np.float32))  
        print("Epoch: {} Validation Loss: {}  Validation Acc: {}".format(epoch, v_loss, v_acc) )

        logger.info('epoch = {} of {} completed, train loss = {:.4f}, val loss = {:.4f}'.format(epoch,args.num_epochs,t_loss, v_loss))
        
        if (epoch+1)%2 == 0:
            ret_states = executeModelonTest(model, args.data_list_val, args.data_dir)
            logger.info('epoch = {}, iou_f = {:.4f}, iou_b = {:.4f}, prec = {:.4f}, rec = {:.4f}, F1 = {:.4f}'.format(epoch,ret_states[0],ret_states[1],ret_states[2],ret_states[3],ret_states[4]))
            mIoU = ret_states[0]
            
            if mIoU > mIoUPrev:
                model.save(osp.join(args.snapshot_dir, 'segModelbestWeights_256_8SU.h5'))
                mIoUPrev = mIoU
                print('Epoch Number: ', epoch)
                print('mean IoU: ', mIoU)
            if (epoch+1)%4 == 0:
                print('current mean IoU: ', mIoU)
                
                if mIoU >mIoUPrev2 and epoch>1:
                    model.save(osp.join(args.snapshot_dir, 'segModelbestWeights_256_8SU_2nd.h5'))
                    mIoUPrev2 = mIoU    

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()