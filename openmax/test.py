import sys
sys.path.append("../")

import os
import csv
import argparse
import numpy as np

import plot
import utils
import resnet
import resnext
import resnet_cifar
import metrics
import dataloader

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax

parser = argparse.ArgumentParser(description='Misclassification Detection / Out of Distribution Detection / Open Set Recognition')

parser.add_argument('--batch-size', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--model', default='res18', type=str, help='model: res18 (default: res18)')
parser.add_argument('--in-data', default='cifar40', type=str, help='in distribution dataset: cifar40 (default: cifar40)')
parser.add_argument('--pos-label', default=0, type=int)
parser.add_argument('--data', default='cifar40', type=str, help='datasets: cifar100 / tinyimagenet / svhn / LSUN (default: cifar100)')
parser.add_argument('--data-root', default='/daintlab/data/md-ood-osr/data/', type=str, help='data path')
parser.add_argument('--model-path', default='../exp-results/', type=str, help='model path')
parser.add_argument('--save-path', default='./exp-results/', type=str, help='save root')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu number')

# openmax
parser.add_argument('--train-class-num', default=40, type=int, help='Classes used in training')

# Parameters for weibull distribution fitting.
parser.add_argument('--weibull-tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull-alpha', default=40, type=int, help='Classes used in testing')
parser.add_argument('--weibull-threshold', default=0.5, type=float, help='Classes used in testing')

# distance
parser.add_argument('--distance', default='euclidean', type=str, help='distance type')
parser.add_argument('--eu-weight', default=5e-3, type=float)

# Architecture
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base-width', type=int, default=64, help='Number of channels in each group.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True

    if args.model == 'res18':
        net = resnet.ResNet18(num_classes=40).cuda()
    elif args.model =='resnext':
        net = resnext.ResNeXt(cardinality=args.cardinality, 
                              depth=args.depth, 
                              nlabels=40, 
                              base_width=args.base_width, 
                              widen_factor=args.widen_factor).cuda()
    elif args.model =='res_cifar':
        net = resnet_cifar.resnet20(num_classes=40).cuda()

    state_dict = torch.load(f'{args.model_path}/model_200.pth')
    net.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss().cuda()
    metric_logger = utils.Logger(os.path.join(args.save_path, 'test_metric.log'))

    ''' Misclassification Detection '''
    print('')
    print('Misclassification Detection')
    print('data: CIFAR40')
    print('')
    
    train_loader = dataloader.train_loader(args.data_root,
                                           args.data,
                                           args.batch_size)

    test_loader, test_targets = dataloader.test_loader(args.data_root,
                                                       args.in_data,
                                                       args.batch_size,
                                                       mode='test')
    
    in_softmax, in_openmax, in_softlogit, in_openlogit, in_open_pred, \
                correct, labels = test(net, train_loader, test_loader)

    acc, auroc, aurc, eaurc, \
    fpr, aupr, ece, li_acc, li_count = metrics.md_metrics_om(in_openlogit,
                                                            in_openmax,
                                                            correct,
                                                            labels)
                                                          

    plot.draw_reliability_diagrams(args.save_path, li_acc, li_count, ece)
    metric_logger.write(['Miscls Detect', '\t\t',
                         'ACCURACY', '\t',
                         'AUROC', '\t\t',
                         'AURC', '\t\t',
                         'E-AURC', '\t\t',
                         'AUPR', '\t\t',
                         'FPR@95%TPR', '\t',
                         'ECE'])
    metric_logger.write(['\t', '\t\t',
                         acc * 100, '\t',
                         auroc * 100, '\t',
                         aurc * 1000, '\t',
                         eaurc * 1000, '\t',
                         aupr * 100, '\t',
                         fpr * 100, '\t',
                         ece * 100])

    with open(f'{args.save_path}/base-scores.csv', 'w', newline='') as f:
        columns = ["",
                   "ACC",
                   "AUROC",
                   "AURC",
                   "E-AURC",
                   "AUPR",
                   "FPR@95%TPR",
                   "ECE"]
        writer = csv.writer(f)
        writer.writerow(['* Misclassification Detection'])
        writer.writerow(columns)
        writer.writerow(
            ['',
             acc * 100,
             auroc * 100,
             aurc * 1000,
             eaurc * 1000,
             aupr * 100,
             fpr * 100,
             ece * 100])
        writer.writerow([''])
    f.close()


    ''' test '''
    print('')
    print('Open Set Recognition-Test')
    print('known data: CIFAR40')
    print('unknown data: CIFAR60')
    print('')
    in_test_loader = dataloader.in_dist_loader(args.data_root,
                                               args.in_data,
                                               args.batch_size,
                                               'test')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'cifar60',
                                                 args.batch_size,
                                                 'test')

    in_softmax, in_openmax, in_softlogit, in_openlogit,\
        _, _, _ = test(net, train_loader, in_test_loader)
    out_softmax, out_openmax, out_softlogit, out_openlogit,\
        _, _, _ = test(net, train_loader, ood_test_loader)
        
        
    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(1-np.array(in_openmax), 1-np.array(out_openmax),
                                              pos_label=0)
    ood_scores = metrics.ood_metrics(1-np.array(in_openmax), 1-np.array(out_openmax))

    metric_logger.write(['TEST CIFAR40-CIFAR60', '\t',
                         'FPR@95%TPR', '\t',
                         'DET ERR', '\t',
                         'AUROC', '\t\t',
                         'AUPR-IN', '\t',
                         'AUPR-OUT', '\t',
                         'F1 SCORE', '\t',
                         ''])
    metric_logger.write(['', '\t\t\t',
                         100 * ood_scores['FPR95'], '\t',
                         100 * ood_scores['DTERR'], '\t',
                         100 * ood_scores['AUROC'], '\t',
                         100 * ood_scores['AUIN'], '\t',
                         100 * ood_scores['AUOUT'], '\t',
                         f1, '\t',
                         ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='CIFAR60',
                 mode='test', task='OsR')


    with open(f'{args.save_path}/base-scores.csv', 'a', newline='') as f:
        columns = ["",
                   "FPR@95%TPR",
                   "DET ERR",
                   "AUROC",
                   "AUPR-IN",
                   "AUPR-OUT",
                   "F1 SCORE"]
        writer = csv.writer(f)
        writer.writerow(['* Open Set Recognition Test-CIFAR60'])
        writer.writerow(columns)
        writer.writerow(
            ['', 100 * ood_scores['FPR95'],
             100 * ood_scores['DTERR'],
             100 * ood_scores['AUROC'],
             100 * ood_scores['AUIN'],
             100 * ood_scores['AUOUT'],
            f1])
        writer.writerow([''])
    f.close()




    ''' Out of Distribution Detection '''
    ''' test '''
    print('')
    print('Out of Distribution Detection-Test')
    print('known data: CIFAR40')
    print('unknown data: SVHN')
    print('')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'svhn',
                                                 args.batch_size,
                                                 'test')

    out_softmax, out_openmax, out_softlogit, out_openlogit,\
        _, _, _ = test(net, train_loader, ood_test_loader)
        
        
    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(1-np.array(in_openmax), 1-np.array(out_openmax),
                                              pos_label=0)
    ood_scores = metrics.ood_metrics(1-np.array(in_openmax), 1-np.array(out_openmax))

    metric_logger.write(['TEST CIFAR40-SVHN', '\t',
                         'FPR@95%TPR', '\t',
                         'DET ERR', '\t',
                         'AUROC', '\t\t',
                         'AUPR-IN', '\t',
                         'AUPR-OUT', '\t',
                         'F1 SCORE', '\t',
                         ''])
    metric_logger.write(['', '\t\t\t',
                         100 * ood_scores['FPR95'], '\t',
                         100 * ood_scores['DTERR'], '\t',
                         100 * ood_scores['AUROC'], '\t',
                         100 * ood_scores['AUIN'], '\t',
                         100 * ood_scores['AUOUT'], '\t',
                         f1, '\t',
                         ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='SVHN',
                 mode='test', task='OoD')


    with open(f'{args.save_path}/base-scores.csv', 'a', newline='') as f:
        columns = ["",
                   "FPR@95%TPR",
                   "DET ERR",
                   "AUROC",
                   "AUPR-IN",
                   "AUPR-OUT",
                   "F1 SCORE"]
        writer = csv.writer(f)
        writer.writerow(['* Out of Distribution Detection Test-SVHN'])
        writer.writerow(columns)
        writer.writerow(
            ['', 100 * ood_scores['FPR95'],
             100 * ood_scores['DTERR'],
             100 * ood_scores['AUROC'],
             100 * ood_scores['AUIN'],
             100 * ood_scores['AUOUT'],
            f1])
        writer.writerow([''])
    f.close()



    print('')
    print('Out of Distribution Detection-Test')
    print('known data: CIFAR40')
    print('unknown data: LSUN-FIX')
    print('')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'lsun-fix',
                                                 args.batch_size,
                                                 'test')

    out_softmax, out_openmax, out_softlogit, out_openlogit,\
        _, _, _ = test(net, train_loader, ood_test_loader)
        
        
    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(1-np.array(in_openmax), 1-np.array(out_openmax),
                                              pos_label=0)
    ood_scores = metrics.ood_metrics(1-np.array(in_openmax), 1-np.array(out_openmax))

    metric_logger.write(['TEST CIFAR40-LSUNFIX', '\t',
                         'FPR@95%TPR', '\t',
                         'DET ERR', '\t',
                         'AUROC', '\t\t',
                         'AUPR-IN', '\t',
                         'AUPR-OUT', '\t',
                         'F1 SCORE', '\t',
                         ''])
    metric_logger.write(['', '\t\t\t',
                         100 * ood_scores['FPR95'], '\t',
                         100 * ood_scores['DTERR'], '\t',
                         100 * ood_scores['AUROC'], '\t',
                         100 * ood_scores['AUIN'], '\t',
                         100 * ood_scores['AUOUT'], '\t',
                         f1, '\t',
                         ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='LSUN-FIX',
                 mode='test', task='OoD')

    with open(f'{args.save_path}/base-scores.csv', 'a', newline='') as f:
        columns = ["",
                   "FPR@95%TPR",
                   "DET ERR",
                   "AUROC",
                   "AUPR-IN",
                   "AUPR-OUT",
                   "F1 SCORE"]
        writer = csv.writer(f)
        writer.writerow(['* Out of Distribution Detection Test-LSUN-FIX'])
        writer.writerow(columns)
        writer.writerow(
            ['', 100 * ood_scores['FPR95'],
             100 * ood_scores['DTERR'],
             100 * ood_scores['AUROC'],
             100 * ood_scores['AUIN'],
             100 * ood_scores['AUOUT'],
            f1])
        writer.writerow([''])
    f.close()



    print('')
    print('Out of Distribution Detection-Test')
    print('known data: CIFAR40')
    print('unknown data: new-TinyImageNet158')
    print('')
    ood_test_loader = dataloader.out_dist_loader(args.data_root,
                                                 'new-tinyimagenet158',
                                                 args.batch_size,
                                                 'test')

    out_softmax, out_openmax, out_softlogit, out_openlogit,\
        _, _, _ = test(net, train_loader, ood_test_loader)
        
        
    f1, li_f1, li_thresholds, \
    li_precision, li_recall = metrics.f1_score(1-np.array(in_openmax), 1-np.array(out_openmax),
                                              pos_label=0)
    ood_scores = metrics.ood_metrics(1-np.array(in_openmax), 1-np.array(out_openmax))

    metric_logger.write(['TEST CIFAR40-Tiny158', '\t',
                         'FPR@95%TPR', '\t',
                         'DET ERR', '\t',
                         'AUROC', '\t\t',
                         'AUPR-IN', '\t',
                         'AUPR-OUT', '\t',
                         'F1 SCORE', '\t',
                         ''])
    metric_logger.write(['', '\t\t\t',
                         100 * ood_scores['FPR95'], '\t',
                         100 * ood_scores['DTERR'], '\t',
                         100 * ood_scores['AUROC'], '\t',
                         100 * ood_scores['AUIN'], '\t',
                         100 * ood_scores['AUOUT'], '\t',
                         f1, '\t',
                         ''])

    plot.draw_f1(args.save_path, f1, li_f1, li_thresholds, data='new-TinyImageNet158',
                 mode='test', task='OoD')

    with open(f'{args.save_path}/base-scores.csv', 'a', newline='') as f:
        columns = ["",
                   "FPR@95%TPR",
                   "DET ERR",
                   "AUROC",
                   "AUPR-IN",
                   "AUPR-OUT",
                   "F1 SCORE"]
        writer = csv.writer(f)
        writer.writerow(['* Out of Distribution Detection Test-new-TinyImageNet158'])
        writer.writerow(columns)
        writer.writerow(
            ['', 100 * ood_scores['FPR95'],
             100 * ood_scores['DTERR'],
             100 * ood_scores['AUROC'],
             100 * ood_scores['AUIN'],
             100 * ood_scores['AUOUT'],
             f1])
        writer.writerow([''])
    f.close()

def test(net, trainloader, testloader):
    net.eval()

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, net, args.eu_weight)
    print("Fittting Weibull distribution...")
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, args.distance)

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    li_softmax, li_openmax = [], []
    li_softlogit, li_openlogit = [], []
    
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         args.eu_weight, args.weibull_alpha, args.distance)  # openmax_prob, softmax_prob
        li_softlogit.append((ss))
        li_openlogit.append((so))
        li_softmax.append(np.max(ss))
        li_openmax.append(so[args.train_class_num])
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
    correct = (labels == pred_openmax)
    
    return np.array(li_softmax), li_openmax, li_softlogit, li_openlogit, pred_openmax, correct, labels

if __name__ == "__main__":
    main()
