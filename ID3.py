from __future__ import print_function
import numpy as np
import pandas as pd
import csv
import random
import time
start = time.time()
class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label

def get_data_label(dataset):
    data = []
    label = []
    for x in dataset:
        data.append(x[:303])
        label.append(x[-1])

    return data, label
def entropy(freq):
    #Tính entropy
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

def load_data(filename):
        lines = csv.reader(open(filename, "r"))
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]

        return dataset

def split_data(dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))

        return [trainSet, copy]
def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

class DecisionTreeID3(object):
    #Hàm __init__ : khởi tạo cây quyết định
    def __init__(self, max_depth= 13, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain
    #Hàm fit : thiết lập hình dạng cho cây quyết định
    def fit(self, data, target):
        self.Ntrain = data.count()[0] #Hàm đếm số lượng mẫu có trong tập dữ liệu huấn luyện
        self.data = data #Tương ứng với X
        self.attributes = list(data) #Chứa các loại thuộc tính điều kiện :outlook,temperature,humidity,wind
        self.target = target #Thuộc tính kết quả(play)
        self.labels = target.unique() #Trả về số lần xuất hiện của các thuộc tính riêng biệt của thuộc tính kết quả
        ids = range(self.Ntrain) #Hàm trả về 1 mảng các giá trị từ 0 đến self.Ntrain
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0) #Hàm tạo nút
        queue = [self.root]
        while queue:
            node = queue.pop() #Lấy giá trị sau cùng của queue
            if node.depth < self.max_depth or node.entropy < self.min_gain: #Kiểm tra điều kiện tạo nút con
                node.children = self._split(node) #Trả về các nút con sau khi đã phân chia dự trên thuộc tính điều kiện
                if not node.children: #node.children là nút lá
                    self._set_label(node) #Thiết lập nhãn
                queue += node.children #Đẩy tất cả các node.children vào queue
            else:
                self._set_label(node) #Thiết lập nhãn

    #Hàm _entropy : khởi tạo và trả về entropy
    def _entropy(self, ids):
        #Tính entropy của nút với các vị trí ids
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids] #Chỉ số bắt đầu từ 1
        freq = np.array(self.target[ids].value_counts()) #Trả vế 1 mảng chứa số lần suất hiện của các giá trị khác nhau trong thuộc tính kết quả
        return entropy(freq)
    #Hàm _set_label: Tìm nhãn(kết quả) cho 1 nút nếu nó là nút lá
    def _set_label(self, node):
        target_ids = [i + 1 for i in node.ids]
        node.set_label(self.target[target_ids].mode()[0]) #Chọn giá trị phổ biến nhất trong nút lá làm nhãn
    #Hàm _split: phân chia các thuộc tính điều kiện vào các nút,
    # tính informatio gain để tìm thuộc tính có độ lợi thông tin tốt nhất,
    # đặt giá trị cho các nút
    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :] #Lấy tất cả các mẫu ở các vị trí ids
        for i, att in enumerate(self.attributes): # enumerate cài đặt mảng có thêm giá trị unique 0 1 2... cho từng giá trị
            values = self.data.iloc[ids, i].unique().tolist() #Lấy các mẫu vị trí ids, ở cột thứ i(ứng với từng thuộc tính điều kiện)
            if len(values) == 1: continue # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist() #Trả về 1 mảng chứa vị trí các mẫu có cùng giá trị thuộc tính điều kiện i
                splits.append([sub_id-1 for sub_id in sub_ids]) #Giảm tất cả các giá trị của sub_ids đi 1 và chèn sub_ids vào splits
            if min(map(len, splits)) < self.min_samples_split: continue #Không phân chia nếu 1 nếu splits có phần tử có chiều dài nhỏ hơn min_samples_split
            # information gain
            HxS= 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids) #Tính H(X,S)
            gain = node.entropy - HxS #Định nghĩa information gain
            if gain < self.min_gain: continue # Dừng lại nếu gain < min_gain
            if gain > best_gain: #Tìm ra gain lớn nhất trong i và chọn nó làm nút kế tiếp
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order) #Đặt giá trị cho nút
        #Tạo các child_nodes với từng nút là các mẫu thuộc best_splits
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes
    #Hàm predict: trả về kết quả dự đoán bằng cách tìm ra nút lá trên cây quyết định
    # dựa vào các giá trị thuộc tính điều kiện
    def predict(self, new_data):
        npoints = new_data.count()[0]
        labels = [None]*npoints #Tạo ra mảng None
        for n in range(npoints):
            # one point
            x = new_data.iloc[n, :]
            # start from root and recursively travel if not meet a leaf
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
                #Tìm ra nút lá dựa theo giá trị của các thuộc tính điều kiện
            labels[n] = node.label
            #Trả về dự đoán dựa trên các thuộc tính điều kiện
        return labels

if __name__ == "__main__":

    # heart.csv = data3.csv
    # Không thuộc tính age, sex..
    filename = 'heart.csv'
    dataset = load_data(filename)

    # Không thuộc tính age, sex..
    testSet = load_data('split_data_test.csv')

    #Có thuộc tính age, sex
    trainingSetData = pd.read_csv('split_data_training_co_thuoc_tinh.csv')  # Hàm load file dữ liệu huấn luyện

    # Có thuộc tính age, sex
    testData = pd.read_csv('split_data_test_co_thuoc_tinh.csv')  # Hàm load file dữ liệu huấn luyện
    #
    X = trainingSetData.iloc[:, :-1]
    y = trainingSetData.iloc[:, -1]

    A = testData.iloc[:, :-1]

    # prepare model
    tree = DecisionTreeID3(max_depth=13, min_samples_split=2)
    tree.fit(X, y)
    print('Data size {0} \nTraining Size = {1} \nTest Size = {2}'.format(len(dataset), len(trainingSetData), len(testData)))

    # test model
    predictions = tree.predict(A)

    # Kiểm tra độ chính xác
    accuracy = get_accuracy(testSet, predictions)
    print('Accuracy of my implement: {0}%'.format(accuracy))

    #Thử nghiệm với sklearn
    # Import Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier

    training = pd.read_csv('split_data_training.csv')
    dataset = pd.read_csv('split_data_test_13_thuoc_tinh.csv')
    dataTrain, labelTrain = get_data_label(training)
    dataTest, labelTest = get_data_label(testSet) # testSet or testData

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # # Train Decision Tree Classifer
    clf = clf.fit(X, y)

    score = clf.score(dataset, labelTest)
    #
    print('Accuracy of sklearn: {0}%'.format(score * 100))
end = time.time()
print('Tine run', end - start)
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus
#
# #split dataset in features and target variable
# feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
#
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols, class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('heart_decision_tree.png')
# Image(graph.create_png())