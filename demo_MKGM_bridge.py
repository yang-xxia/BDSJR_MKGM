import time
import argparse
from engine_rcnn_mamba import *
from bridge import *
from models import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 假设你想使用第二个GPU（从0开始计数）
import tensorboard
parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', metavar='DIR',default='dataset',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_mkgm():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Classification(args.data, 'trainval', inp_name='dataset/ent_emb.pkl')
    val_dataset = Classification(args.data, 'test', inp_name='dataset/ent_emb.pkl')

    num_classes = 29

    # load model
    model = rcnn_mamba()
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    # Optional: Print parameters' shapes
    for name, param in model.named_parameters():
        print(f'{name}: {param.shape}')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/MKGM/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    
    # Start timing the inference
    start_time = time.time()

    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

    # End timing the inference
    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Total inference time: {inference_time:.2f} seconds')

    # Calculate and print the inference speed
    total_samples = len(val_dataset)  # Assuming val_dataset has the total number of samples
    inference_speed = total_samples / inference_time  # samples per second
    print(f'Inference speed: {inference_speed:.2f} samples per second')


if __name__ == '__main__':
    main_mkgm()
