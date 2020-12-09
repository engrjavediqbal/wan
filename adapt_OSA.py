import argparse
import numpy as np
import os
import os.path as osp
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from unet.model import unet_Model, discriminator_OSA
import timeit
import util
from keras.layers import Input
from keras.models import Model, load_model
import keras.optimizers as optimizers

start = timeit.default_timer()

BATCH_SIZE = 4
SRC_DATA_DIRECTORY = './datasets/rwanda'
SRC_DATA_LIST_PATH_TRAIN = './datasets/dataLists/rwanda/train.txt'
SRC_DATA_LIST_PATH_VAL = './datasets/dataLists/rwanda/val.txt'

TGT_DATA_DIRECTORY = './datasets/isprsPotsdam'
TGT_DATA_LIST_PATH_TRAIN = './datasets/dataLists/isprsPotsdam/train.txt'
TGT_DATA_LIST_PATH_VAL = './datasets/dataLists/isprsPotsdam/test.txt'



INPUT_SIZE = '256,256'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
# SNAPSHOT_DIR = './snapshots/rwanda/src_train/'
SNAPSHOT_DIR = './snapshots/isprsPotsdam/rwanda2isprsPotsdam/OSA/'
WEIGHT_DECAY = 1e-6
MODEL = 'Unet'
LOG_FILE = 'log'
SRC_TRAINED_MODEL = './snapshots/rwanda/src_train/segModelbestWeights_256_8SU.h5'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=SRC_DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list-train", type=str, default=SRC_DATA_LIST_PATH_TRAIN,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-list-val", type=str, default=SRC_DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--tgt-data-dir", type=str, default=TGT_DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--tgt-data-list-train", type=str, default=TGT_DATA_LIST_PATH_TRAIN,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--tgt-data-list-val", type=str, default=TGT_DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--src-trained-model", type=str, default=SRC_TRAINED_MODEL,
                        help="Path to the source dataset trained model")
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

def load_imagesDisc(train_indices, batch_size, itr, lines, image_size, stFlag, data_dir):

    # Create empty arrays to contain batch of features and labels#
    batch_images = []
    batch_labels = []
    startInd = itr*batch_size
    endInd = startInd + batch_size
    for i in range(startInd, endInd):
        line = lines[train_indices[i]].split(' ')

        img = cv2.imread(osp.join(data_dir, line[0]))

        img = img[...,::-1]
    
        if stFlag==1:
            binLabel = np.ones(image_size)
        else:
            binLabel = np.zeros(image_size)
        batch_images.append(img)
        binLabel = np.reshape(binLabel, (image_size[0], image_size[1],1))
        batch_labels.append(binLabel)
    batch_images = np.array(batch_images, dtype=np.float32)
    batch_labels = np.array(batch_labels)
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


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def  GANS(image_size, generatorModel, discrimnatorModel):

    inputSeg = Input((image_size[0], image_size[1], 3))
    inputs = Input((image_size[0], image_size[1], 3))
    
    outPutSeg = generatorModel(inputSeg)
    
    H = generatorModel(inputs)
    discrimnatorModel.trainable = False
    gan_V = discrimnatorModel(H)
    GAN = Model([inputSeg, inputs], [outPutSeg, gan_V])
    opt = optimizers.Adam(lr=0.000001, decay=1e-6)
    GAN.compile(loss=['binary_crossentropy', 'binary_crossentropy'], 
        optimizer=opt,
        metrics = ['accuracy'], loss_weights=[1.0, 0.10]) 
    return GAN


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    logger = util.set_logger(args.snapshot_dir, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    h, w = map(int, args.input_size.split(','))
    image_size = (h, w)

    # Create network.

    generatorModel = unet_Model(image_size, args.learning_rate, args.weight_decay)
    generatorModel.load_weights(args.src_trained_model)

    discrimnatorModel = discriminator_OSA(image_size)

    Gan  = GANS(image_size, generatorModel, discrimnatorModel)


    lines = [line.rstrip('\n') for line in open(args.data_list_train)]
    lines1 = [line.rstrip('\n') for line in open(args.data_list_val)]
    linesTar = [line.rstrip('\n') for line in open(args.tgt_data_list_train)]
    linesTarVal = [line.rstrip('\n') for line in open(args.tgt_data_list_val)]



    indicesTrain = np.arange(len(lines))
    indicesVal = np.arange(len(lines1))
    indicesTrainTar = np.arange(len(linesTar))
    indicesValTar = np.arange(len(linesTarVal))


    print(indicesTrain.shape)
    print(indicesVal.shape)

    np.random.shuffle(indicesTrainTar)


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

        d_loss = []
        d_acc = []

        if epoch < 2:

            for i in range(np.int(len(indicesTrainTar)/args.batch_size)):
#         for i in range(2):
                xs = None
                ys = None

                Xs_train, ys_train = load_imagesDisc(indicesTrain, args.batch_size, i, lines, image_size, 1, args.data_dir)
                Xt_train, yt_train = load_imagesDisc(indicesTrainTar, args.batch_size, i, linesTar, image_size, 0, args.tgt_data_dir)

                xs = np.concatenate((Xs_train, Xt_train), axis=0)
                ys = np.concatenate((ys_train, yt_train), axis=0)

                xs = np.array(xs)
                ys = np.array(ys)

                xs = generatorModel.predict(xs)
                xs = np.array(xs)

                histD = discrimnatorModel.train_on_batch(xs,ys)
                print(histD)
                d_loss.append(histD[0])
                d_acc.append(histD[1])
        else:
            for i in range(np.int(len(indicesTrainTar)/args.batch_size)):
                xs = None
                ys = None
                Xs_train, ys_train = load_imagesDisc(indicesTrain, args.batch_size, i, lines, image_size, 1, args.data_dir)
                Xt_train, yt_train = load_imagesDisc(indicesTrainTar, args.batch_size, i, linesTar, image_size, 0, args.tgt_data_dir)
        #         print(X_train.shape)


                xs = np.concatenate((Xs_train, Xt_train), axis=0)
                ys = np.concatenate((ys_train, yt_train), axis=0)

                xs = np.array(xs)
                ys = np.array(ys)

                xs = generatorModel.predict(xs)
                xs = np.array(xs)

                histD = discrimnatorModel.train_on_batch(xs,ys)
    #             print(histD)
                d_loss.append(histD[0])
                d_acc.append(histD[1])
                
                Xt_train, yt_train = load_imagesDisc(indicesTrainTar, args.batch_size, i, linesTar, image_size, 1, args.data_dir)
                Xs_train, ys_train = load_images(indicesTrain, args.batch_size, i, lines, image_size[0], args.data_dir)
                
    #             Gan.compile(loss=discrimnatorModel.loss, optimizer= discrimnatorModel.optimizer , metrics=discrimnatorModel.metrics )

                history = Gan.train_on_batch([Xs_train, Xt_train], [ys_train, yt_train])
                t_loss.append(history[1])
                t_acc.append(history[2])

        t_loss = np.mean(np.array(t_loss, dtype=np.float32))
        t_acc = np.mean(np.array(t_acc, dtype=np.float32))  
        d_loss = np.mean(np.array(d_loss, dtype=np.float32))
        d_acc = np.mean(np.array(d_acc, dtype=np.float32))

        # print("Epoch: {}  Disc_Loss: {:0.4f}  Disc_Acc: {:0.4f} Seg_Loss: {:0.4f}  Adv_Loss: {:0.4f}".format(epoch, d_loss, d_acc, t_loss, t_acc) )
 
        logger.info('epoch = {} of {}, Disc_Loss: {:0.4f}  Disc_Acc: {:0.4f} Seg_Loss: {:0.4f}  Adv_Loss: {:0.4f}'.format(epoch,args.num_epochs,d_loss, d_acc, t_loss, t_acc))
        
        if (epoch+1)%2 == 0:
            ret_states = executeModelonTest(generatorModel, args.tgt_data_list_val, args.tgt_data_dir)
            logger.info('epoch = {}, iou_f = {:.4f}, iou_b = {:.4f}, prec = {:.4f}, rec = {:.4f}, F1 = {:.4f}'.format(epoch,ret_states[0],ret_states[1],ret_states[2],ret_states[3],ret_states[4]))
            mIoU = ret_states[0]
            
            if mIoU > mIoUPrev:
                generatorModel.save(osp.join(args.snapshot_dir, 'genModelbestWeights.h5'))
                discrimnatorModel.save(osp.join(args.snapshot_dir, 'descModelbestWeights.h5'))
                mIoUPrev = mIoU
                print('Epoch Number: ', epoch)
                print('mean IoU: ', mIoU)
            if (epoch+1)%4 == 0:
                print('current mean IoU: ', mIoU)
                
                if mIoU >mIoUPrev2 and epoch>1:
                    generatorModel.save(osp.join(args.snapshot_dir, 'genModelbestWeights_2nd.h5'))
                    discrimnatorModel.save(osp.join(args.snapshot_dir, 'descModelbestWeights_2nd.h5'))
                    mIoUPrev2 = mIoU    

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()