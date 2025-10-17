# NLP（IMDB情感分类）

使用经典的 IMDB 影评数据集来完成情感分类任务。
IMDB 影评数据集包含了50000 条用户评价，评价的标签分为消极和积极， 其中 IMDB 评级<5 的用户评价标注为0，即消极； IMDB 评价>=7 的用户评价标注为 1，即积极。 25000条影评用于训练集，25000 条用于测试集。

## 一、加载数据

使用tensorflow.keras.dataset加载数据集

```py
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
```

第一次加载会下载数据集到`C:\Users\name\.keras\datasets`目录下  
加载数字编码表

```py
keras.datasets.imdb.get_word_index()
```

这是一个单词表，每个词对应唯一的数字用于查询，对表进行处理并翻转键值对，通过数字索引单词  

```py
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = {value:key for (key, value) in word_index.items()}
```

使用`pad_sequences`对文本进行截取，保证输入词序列长度一致

```py
keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
```

对数据进行打乱并分批

```py
db_data = tf.data.Dataset.from_tensor_slices((pad_x_train,y_train)).shuffle(1000)
db_train = db_train.batch(batchsz,drop_remainder=True)
```

## 二、模型训练

### 1、构建模型

使用一个词嵌入层，将文本转化为词向量作为输入，使用双向LSTM进行编码，分类器使用两层全连接网络（32 unit+1 unit）  
优化器使用adam，损失函数使用二元交叉熵

```py
class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.embedding = Sequential([
            layers.Embedding(total_words,embedding_len,
                             input_length=max_review_len)
        ])
        self.rnn = Sequential([
            layers.Bidirectional(layers.LSTM(units,dropout=0.3))
        ])

        self.outlayer = Sequential([
            layers.Dense(32,activation='relu'),
            layers.Dense(1,activation='sigmoid')
        ])

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)

        x = self.outlayer(x,training)
        return x

units = 32
epochs = 5

model = MyRNN(units)
model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])
model.build(input_shape=(None,max_review_len))
model.summary()
```

### 2、训练

使用tensorboard可以将训练过程可视化

```py
from tensorflow.keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir="./log")
```

训练模型

```py
history = model.fit(db_train, epochs=epochs, validation_data=db_val, callbacks=tbCallBack)
```

```text
Epoch 1/5
156/156 [==============================] - 12s 59ms/step - loss: 0.6106 - accuracy: 0.6274 - val_loss: 0.3041 - val_accuracy: 0.8774
Epoch 2/5
156/156 [==============================] - 8s 49ms/step - loss: 0.2622 - accuracy: 0.8933 - val_loss: 0.2834 - val_accuracy: 0.8936
Epoch 3/5
156/156 [==============================] - 8s 48ms/step - loss: 0.1773 - accuracy: 0.9365 - val_loss: 0.3647 - val_accuracy: 0.8702
Epoch 4/5
156/156 [==============================] - 7s 48ms/step - loss: 0.1406 - accuracy: 0.9501 - val_loss: 0.3331 - val_accuracy: 0.8870
Epoch 5/5
156/156 [==============================] - 8s 50ms/step - loss: 0.1161 - accuracy: 0.9582 - val_loss: 0.3512 - val_accuracy: 0.8862
```

模型训练5轮后验证集上准确率88.62%，测试集准确率为86.34%

```text
195/195 [==============================] - 3s 16ms/step - loss: 0.4208 - accuracy: 0.8634
```

## 三、模型保存

```py
import os
os.makedirs('models',exist_ok=True)
model.save_weights('models/imdb_weights.h5')
```
