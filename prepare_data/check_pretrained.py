import os


def main():
    root_path = '/wdata'
    weights_dir = 'train_logs'
    weights_file = 'epoch_10.pth'
    link_to_pretrained = 'https://www.dropbox.com/s/agchus2n6pe5uyn/epoch_10.pth?dl=1'
    
    path = '/wdata'
    pretrained = 'pretrained'
    pretrained_path = os.path.join(path, pretrained)
    
    os.mkdir(pretrained_path)
    
    mxnet_log = 'https://www.dropbox.com/s/3ql6rbjr14pf2mk/log?dl=1'
    mxnet_model = 'https://www.dropbox.com/s/5p32ipi65069z79/model-0000.params?dl=1'
    mxnet_symbol = 'https://www.dropbox.com/s/yb3cuo780phyomo/model-symbol.json?dl=1'

    knn_model = 'https://www.dropbox.com/s/z223mqm0as4uiqx/model_original_1.pkl?dl=1'
    subpath = os.path.join(root_path, weights_dir)
    if os.path.exists(subpath):
        weights_path = os.path.join(subpath, weights_file)
        if os.path.exists(weights_path):
            print('Weights exists')
        else:
            print('Clear folder and try again')
    else:
        weights_path = os.path.join(subpath, weights_file)
        os.mkdir(subpath)
        load_commnad = 'wget {} -O {}'.format(link_to_pretrained, weights_path)
        os.system(load_commnad)
        print('Home pretrained weights loaded!')


        tmp_save_path = os.path.join(pretrained_path, mxnet_log.split('/')[-1].split('?')[0])
        load_commnad = 'wget {} -O {}'.format(mxnet_log, tmp_save_path)
        print(load_commnad)
        os.system(load_commnad)
        tmp_save_path = os.path.join(pretrained_path, mxnet_model.split('/')[-1].split('?')[0])
        load_commnad = 'wget {} -O {}'.format(mxnet_model, tmp_save_path)
        os.system(load_commnad)
        
        tmp_save_path = os.path.join(pretrained_path, mxnet_symbol.split('/')[-1].split('?')[0])
        load_commnad = 'wget {} -O {}'.format(mxnet_symbol, tmp_save_path)
        os.system(load_commnad)

        tmp_save_path = os.path.join(pretrained_path, knn_model.split('/')[-1].split('?')[0])
        load_commnad = 'wget {} -O {}'.format(knn_model, tmp_save_path)
        os.system(load_commnad)

        print('Insigth face loaded')
       


if __name__ == '__main__':
    main()