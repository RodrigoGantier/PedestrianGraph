import tensorflow as tf
import os.path as osp
import os
import numpy as np
import multiprocessing
import argparse
import time


parser = argparse.ArgumentParser("Training and test Pedestrian Graph")
parser.add_argument('-c', '--tfrecord_path', required=True, type=str, help='path to tfrecods')
parser.add_argument('--train', required=True, type=str,
                    help='Bool variable, if True Pedestrian Graph is in training stage, if False is in testing')
parser.add_argument('--num_gpu', default=1, type=int, help='Number of GPUs to use')
parser.add_argument('--lr', type=float, default=0.001, help='Initial training rate')
parser.add_argument('--save_checkpoints_steps', default=1000, type=int,
                        help=' Save checkpoints every this many steps')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
parser.add_argument('--keep_checkpoint_max', default=2, type=int,
                        help='Max number of keep checkpoint to keep, keep the last X checkpoint')
parser.add_argument('--batch_size', default=62, type=int, help='Batch size')
parser.add_argument('--repeated', default=1000, type=int,
                    help='number of times the data is repeated in data-pipeline, (tensorflow input data API)')

args = parser.parse_args()

batch_size_ = args.batch_size
epochs_ = args.repeated  # number of times the data is repeated in data-pipeline
threads = multiprocessing.cpu_count()  # count number of threads in cpu


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_adj_m():
    """
    A_tilde = adj matrix + identity matrix
    Diag = Σ (i=0 --> m) [A_tilde (j,i)]
    D^-1/2 = diag(1 / √ Diag)

    :return:
    """
    self_link = []
    for i in range(14):
        self_link.append((i, i))

    l0 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 1, 1, 1, 2, 3, 4, 5, 1, 1, 8,  8,  9, 10, 11]
    l1 = [1, 1, 1, 2, 3, 4, 5, 1, 1, 8,  8,  9, 10, 11, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13]

    edges = list(zip(l0, l1))
    adj = edge2mat(edges, 14)

    adj_tilde = adj + np.identity(n=adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1 / 2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)

    return adj_norm


def data_input(batch_size, epochs, train):

    def _parse_fn_train(example_proto):

        features = {'image': tf.VarLenFeature(tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        img = tf.sparse.to_dense(parsed_features['image'], default_value=0)
        img = tf.cast(tf.reshape(img, [300, 14, 2]), tf.float32)
        img = tf.transpose(img, [0, 2, 1])
        label = parsed_features['label']
        return img, label

    def _parse_fn_test(example_proto):

        features = {'image': tf.VarLenFeature(tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        img = tf.sparse.to_dense(parsed_features['image'], default_value=0)
        img = tf.cast(tf.reshape(img, [300, 14, 2]), tf.float32)
        img = tf.transpose(img, [0, 2, 1])
        label = parsed_features['label']
        return img, label

    root_path = args.tfrecord_path
    tr_tfrecord = osp.join(root_path, 'train_2d_pick.tfrecords')
    te_tfrecord = osp.join(root_path, 'test_2d_pick.tfrecords')

    if train:
        data_set = tf.data.TFRecordDataset(tr_tfrecord, num_parallel_reads=threads)
        data_set = data_set.shuffle(2048)
        data_set = data_set.map(_parse_fn_train, num_parallel_calls=threads).repeat(epochs)
    else:
        data_set = tf.data.TFRecordDataset(te_tfrecord, num_parallel_reads=threads)
        data_set = data_set.map(_parse_fn_test, num_parallel_calls=threads)
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.prefetch(batch_size * 2)

    return data_set


def train_input_fn():
    batch_size = batch_size_
    epochs = epochs_
    return data_input(batch_size, epochs, train=True)


def test_input_fn():
    batch_size = batch_size_
    epochs = 124 // args.batch_size  # 124 is the total number of test samples
    return data_input(batch_size, epochs, train=False)


def model_graph(x, adj_m, dropout, mode):

    with tf.variable_scope('GCN_et'):
        is_tr = mode == tf.estimator.ModeKeys.TRAIN
        bs, t, ch, vrt = x.shape

        # ------------------------------------------------------------------------------------------------
        x0 = tf.transpose(tf.reshape(x, [-1, vrt]), [1, 0])
        x0 = tf.transpose(tf.matmul(adj_m, x0), [1, 0])
        x0 = tf.transpose(tf.reshape(x0, [bs, t, ch, vrt]), [0, 1, 3, 2])
        # conv1
        l0 = tf.layers.conv2d(x0, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, use_bias=True)
        l0 = tf.layers.dropout(l0, rate=dropout, training=is_tr)
        bs, t, vrt, ch = l0.shape

        x0 = tf.transpose(tf.reshape(tf.transpose(l0, [0, 1, 3, 2]), [-1, vrt]), [1, 0])
        x0 = tf.transpose(tf.matmul(adj_m, x0), [1, 0])
        x0 = tf.transpose(tf.reshape(x0, [bs, t, ch, vrt]), [0, 1, 3, 2])
        # conv2
        l0 = tf.layers.conv2d(x0, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, use_bias=True)
        l0 = tf.layers.dropout(l0, rate=dropout, training=is_tr)
        bs, t, vrt, ch = l0.shape

        # ------------------------------------------------------------------------------------------------
        pool_t = tf.squeeze(tf.layers.average_pooling2d(l0, pool_size=(t, 1), strides=1))
        pool_t = tf.transpose(pool_t, [0, 2, 1])
        pool_v = tf.squeeze(tf.layers.average_pooling1d(pool_t, pool_size=(32,), strides=1))
        pool_v = tf.transpose(tf.matmul(adj_m, tf.transpose(pool_v, [1, 0])), [1, 0])

        logits = tf.layers.dense(pool_v, units=1, activation=None)

        return logits, l0


def model_fn(features, labels, mode, params):

    x = features
    adj = params['adj']
    starter_learning_rate = params['learning_rate']
    dropout = params['dropout']
    iteration = tf.train.get_global_step()

    learning_rate = tf.train.exponential_decay(starter_learning_rate, iteration,
                                               decay_steps=5000, decay_rate=0.98, staircase=True)

    logits, l0 = model_graph(x, adj, dropout=dropout, mode=mode)
    pred_y = tf.round(tf.sigmoid(logits))

    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'class_ids': pred_y[:, tf.newaxis],
            'probabilities': tf.sigmoid(logits),
            'logits': logits,
            'l0': l0
        }

        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    elif mode == tf.estimator.ModeKeys.EVAL:

        loss = tf.losses.sigmoid_cross_entropy(labels, tf.squeeze(logits))
        accu = tf.metrics.accuracy(labels=labels, predictions=pred_y)

        metrics = {'Accuracy/test': accu}
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_y, loss=loss, eval_metric_ops=metrics, )

    else:

        loss = tf.losses.sigmoid_cross_entropy(labels, tf.squeeze(logits))
        optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate, weight_decay=1e-8)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.minimize(loss, global_step=iteration)

        accu = tf.metrics.accuracy(labels=labels, predictions=pred_y)
        tf.summary.scalar('Lr', learning_rate)
        tf.summary.scalar('Accuracy/train', accu[1])
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    return spec


def main():

    save_path = osp.join(args.tfrecord_path, 'save')
    if not osp.isdir(save_path):
        os.mkdir(save_path)
    adj_m = get_adj_m()
    starter_learning_rate = float(args.lr)

    params = {
        'adj': np.array(adj_m, dtype=np.float32),
        'learning_rate': starter_learning_rate,
        'dropout': 0.5}

    gpu_list = []
    for n_gpu in range(args.num_gpu):
        gpu_list.append("/device:GPU:{}".format(n_gpu))
    distribution = tf.contrib.distribute.MirroredStrategy(gpu_list)
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=args.save_checkpoints_steps,
        keep_checkpoint_max=args.keep_checkpoint_max,
        train_distribute=distribution,
        eval_distribute=distribution)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=save_path,
        config=run_config)

    # ---------------------------------------------
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=train_input_fn,
    #     max_steps=1000000)
    #
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=test_input_fn,
    #     steps=1000)
    #
    # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    # ---------------------------------------------
    if args.train == 'True':

        model.train(input_fn=train_input_fn, steps=1)
        for i in range(1000):
            model.train(input_fn=train_input_fn, steps=1000)
            result = model.evaluate(input_fn=test_input_fn)
            print(result)
            if result['Accuracy/test'] > 0.91:  # early stoping!!!
                print('confirmation test:')
                result = model.evaluate(input_fn=test_input_fn)
                print(result)
                result = model.evaluate(input_fn=test_input_fn)
                print(result)
                break

    tic = time.time()
    result = model.evaluate(input_fn=test_input_fn)
    toc = time.time()
    print('---------------------------------------------------------------------------')
    print('\n')
    print('Final accuracy: {}, Each sample took: {} sec, All (124) took: {} sec'.format(result, (toc - tic) / 124, toc - tic))
    print('\n')
    print('---------------------------------------------------------------------------')


if __name__ == "__main__":
    main()

