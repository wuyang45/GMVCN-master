#coding: utf8
from dataset import *
from random import shuffle


def tree2edge_index(treefile, idx_map):
    
    first = []
    second = []
    with open(treefile, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip()
            father = row.split()[0]
            child = row.split()[1]
            
            if int(idx_map.get(father, -1)) >= 0 and int(idx_map.get(child, -1)) >= 0:
                first.append(idx_map[father])
                second.append(idx_map[child])
            else:
                continue
            
    top_to_down = [first, second]
    
    bottom_to_up = [second, first]
    
    #print(top_to_down)
    #print(bottom_to_up)
    
    return np.array(top_to_down), np.array(bottom_to_up)
    
    
    
def get_tf_idf(all_text):
    corpus = []
    all_tweet_id = []
    with open(all_text, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tweet_id = line.strip().split('|')[0]
            text = line.strip().split('|')[1]
            all_tweet_id.append(tweet_id)
            corpus.append(text)
            
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) #word frequency
    all_word = vectorizer.get_feature_names() #All keywords in bag of words，and the order is useful
    all_word_dict = {i: j for i, j in enumerate(all_word)}
    tfidf = TfidfTransformer().fit_transform(X) 
    all_tweet_idx = {j: i for i, j in enumerate(all_tweet_id)}
    return all_word_dict, tfidf, all_tweet_idx
    
    
    
def generate_tfidf_npz(path, data_name, all_word_dict, tfidf, all_tweet_idx):
    file_path = os.path.join(path, data_name)
    file_list = os.listdir(file_path)
    max_len = 32  
    
    tree_path = os.path.join(path, 'train_tree')
    
    for file in file_list: 
        print(file)
        #each_graph_sample
        
        file_id = file.split('.')[0]
        
        each_sample_ids = []
        x_x = []
        x_user = []
        x_stance = []
            
        with open(os.path.join(file_path, file), 'r', encoding = 'utf-8') as f: #each_graph_sample
            lines = f.readlines()
            
            for line in lines:
                tweet_id = line.split('|')[0]
                text = line.split('|')[1]
                user_f = line.split('|')[2]
                stance = line.split('|')[3]
                label = line.strip().split('|')[-1]
                
                tfidf_idx = all_tweet_idx[tweet_id]  #The subscript of the tweet in tfidf
                tfidf_tid = tfidf[tfidf_idx].toarray() #Representation of the tweet in tfidf
                text_f = tfidf_tid.ravel()[:5000]
                
                user = user_f.split()
                user = list(map(int, user))
                #print(user)
                
                if stance == 'support':
                    stance_label = 0
                elif stance == 'query':
                    stance_label = 1
                elif stance == 'deny':
                    stance_label = 2
                else:
                    stance_label = 3
                    
                if label == 'true':
                    rumor_label = 0
                elif label == 'false':
                    rumor_label = 1
                else:
                    rumor_label = 2
                    
                
                each_sample_ids.append(tweet_id)
                x_x.append(text_f)
                x_user.append(user)
                x_stance.append(stance_label)
                
        idx_map = {j: i for i, j in enumerate(each_sample_ids)} #map to 0-each_sample_num
        
        tree_file = os.path.join(tree_path, file)
        TD_edge_idx, BU_edge_idx = tree2edge_index(tree_file, idx_map)
        x_x, x_user, x_stance, y = np.array(x_x), np.array(x_user), np.array(x_stance), np.array(rumor_label)
        
        np.savez(os.path.join(path, 'train_tfidfNPZ', file_id+'.npz'), x=x_x, x_user= x_user, x_stance=x_stance, edge_index=TD_edge_idx,
                BU_edge_index=BU_edge_idx, y=y)
                

def treeLen(path):
    
    fold_list = ['train', 'eval', 'test']
    
    treeLendic = {}
    
    for fold in fold_list:
        file_list = os.listdir(os.path.join(path, fold))
        #file_list.remove('graph')
        #file_list.remove('one_hot_graph')
        for file in file_list:
            file_path = os.path.join(path, fold, file)
            file_id = file.split('.')[0]
            
            with open(file_path, 'r', encoding = 'utf-8') as f:
                lines = f.readlines()
                Len = len(lines)
                
            treeLendic[file_id] = Len

    return treeLendic 
    
    
def loadData(treeLenDic, path, dropedge, lower):
    train_root = os.path.join(path, 'train_tfidfNPZ')
    train_file_list = os.listdir(train_root)
    #print("loading train set", )
    traindata_list = GraphDataset(train_root, train_file_list, treeLenDic, dropedge, lower)
    #print("train no:", len(traindata_list))
    
    eval_root = os.path.join(path, 'eval_tfidfNPZ')
    eval_file_list = os.listdir(eval_root)
    #print("loading eval set", )
    evaldata_list = GraphDataset(eval_root, eval_file_list, treeLenDic, lower = lower)
    #print("eval no:", len(evaldata_list))
    
    test_root = os.path.join(path, 'test_tfidfNPZ')
    test_file_list = os.listdir(test_root)
    #print("loading test set", )
    testdata_list = GraphDataset(test_root, test_file_list, treeLenDic, lower = lower)
    #print("test no:", len(testdata_list))
    
    return traindata_list, evaldata_list, testdata_list



def treeLenPheme(path):
    
    fold_list = ['charliehebdo-all-rnr-threads', 'ebola-essien-all-rnr-threads', 'ferguson-all-rnr-threads', 'germanwings-crash-all-rnr-threads',
                'gurlitt-all-rnr-threads', 'ottawashooting-all-rnr-threads', 'prince-toronto-all-rnr-threads', 'putinmissing-all-rnr-threads',
                'sydneysiege-all-rnr-threads']
    
    treeLendic = {}
    
    for fold in fold_list:
        file_list = os.listdir(os.path.join(path, fold, 'tree'))

        for file in file_list:
            file_path = os.path.join(path, fold, 'tree', file)
            file_id = file.split('.')[0]
            
            with open(file_path, 'r', encoding = 'utf-8') as f:
                lines = f.readlines()
                Len = len(lines)
                
            treeLendic[file_id] = Len

    return treeLendic 


def loadDataPheme(datapath, treeLenDic, fold_train, fold_test, droprate, lower):
    
    print("loading train set", )
    traindata_list = GraphDataset(datapath, fold_train, treeLenDic, droprate, lower=lower)
    print("train no:", len(traindata_list))
    
    print("loading test set", )
    testdata_list = GraphDataset(datapath, fold_test, treeLenDic, lower=lower)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list
    
    
def get9folddata(path):
    event_list = os.listdir(path)
    
    event0_list = np.array(os.listdir(os.path.join(path, event_list[0])))
    event1_list = np.array(os.listdir(os.path.join(path, event_list[1])))
    event2_list = np.array(os.listdir(os.path.join(path, event_list[2])))
    event3_list = np.array(os.listdir(os.path.join(path, event_list[3])))
    event4_list = np.array(os.listdir(os.path.join(path, event_list[4])))
    event5_list = np.array(os.listdir(os.path.join(path, event_list[5])))
    event6_list = np.array(os.listdir(os.path.join(path, event_list[6])))
    event7_list = np.array(os.listdir(os.path.join(path, event_list[7])))
    event8_list = np.array(os.listdir(os.path.join(path, event_list[8])))
    
    fold0_train = list(np.concatenate((event1_list, event2_list, event3_list, event4_list, event5_list, event6_list, event7_list, event8_list)))
    fold0_test = list(event0_list)
    shuffle(fold0_train)
    shuffle(fold0_test)
    
    fold1_train = list(np.concatenate((event0_list, event2_list, event3_list, event4_list, event5_list, event6_list, event7_list, event8_list)))
    fold1_test = list(event1_list)
    shuffle(fold1_train)
    shuffle(fold1_test)
    
    fold2_train = list(np.concatenate((event0_list, event1_list, event3_list, event4_list, event5_list, event6_list, event7_list, event8_list)))
    fold2_test = list(event2_list)
    shuffle(fold2_train)
    shuffle(fold2_test)
    
    fold3_train = list(np.concatenate((event0_list, event1_list, event2_list, event4_list, event5_list, event6_list, event7_list, event8_list)))
    fold3_test = list(event3_list)
    shuffle(fold3_train)
    shuffle(fold3_test)
    
    fold4_train = list(np.concatenate((event0_list, event1_list, event2_list, event3_list, event5_list, event6_list, event7_list, event8_list)))
    fold4_test = list(event4_list)
    shuffle(fold4_train)
    shuffle(fold4_test)
    
    fold5_train = list(np.concatenate((event0_list, event1_list, event2_list, event3_list, event4_list, event6_list, event7_list, event8_list)))
    fold5_test = list(event5_list)
    shuffle(fold5_train)
    shuffle(fold5_test)
    
    fold6_train = list(np.concatenate((event0_list, event1_list, event2_list, event3_list, event4_list, event5_list, event7_list, event8_list)))
    fold6_test = list(event6_list)
    shuffle(fold6_train)
    shuffle(fold6_test)
    
    fold7_train = list(np.concatenate((event0_list, event1_list, event2_list, event3_list, event4_list, event5_list, event6_list, event8_list)))
    fold7_test = list(event7_list)
    shuffle(fold7_train)
    shuffle(fold7_test)
    
    fold8_train = list(np.concatenate((event0_list, event1_list, event2_list, event3_list, event4_list, event5_list, event6_list, event7_list)))
    fold8_test = list(event8_list)
    shuffle(fold8_train)
    shuffle(fold8_test)
    
    return list(fold0_train), list(fold0_test),\
            list(fold1_train), list(fold1_test),\
            list(fold2_train), list(fold2_test),\
            list(fold3_train), list(fold3_test),\
            list(fold4_train), list(fold4_test),\
            list(fold5_train), list(fold5_test),\
            list(fold6_train), list(fold6_test),\
            list(fold7_train), list(fold7_test),\
            list(fold8_train), list(fold8_test),\
            
            

if __name__ == '__main__':
    
    path = '/workspace/rumor/task3/data/semeval2017'
    
    all_text = os.path.join(path, 'all_text.txt')

    all_word_dict, tfidf, all_tweet_idx = get_tf_idf(all_text)
    
    generate_tfidf_npz(path, 'train', all_word_dict, tfidf, all_tweet_idx)
    