#coding=utf-8
from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import os
import  numpy as np

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
from metrics import clustering_metrics
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iteration =settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']

    def erun(self):
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        # d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])
        # 不要d_real
        _, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        # opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])
        # 修改d_real
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], feas['num_nodes'])


        # Initialize session
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # 自己定义writer
        writer = tf.summary.FileWriter('cnlogs/rpr_vae++/citeseer', sess.graph)

        # Train model
        for epoch in range(self.iteration):
            emb, _ = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

            if (epoch+1) % 2 == 0:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(emb)
                print("Epoch:", '%04d' % (epoch + 1))
                predict_labels = kmeans.predict(emb)
                cm = clustering_metrics(feas['true_labels'], predict_labels)
                cm.evaluationClusterModelFromLabel()

                # 修改
                # cm.evaluationClusterModelFromLabel()
                acc_0, nmi_0, adjscore_0, f1_macro_0, precision_macro_0, recall_macro_0, f1_micro_0, precision_micro_0, recall_micro_0 = cm.evaluationClusterModelFromLabel()
                # 自己写验证
                acc_summary = tf.Summary(value=[tf.Summary.Value(tag="v1/acc", simple_value=acc_0)])
                nmi_summary = tf.Summary(value=[tf.Summary.Value(tag="v1/nmi", simple_value=nmi_0)])
                adjscore_summary = tf.Summary(value=[tf.Summary.Value(tag="v1/adjscore", simple_value=adjscore_0)])
                f1_macro = tf.Summary(value=[tf.Summary.Value(tag="v2/f1_macro", simple_value=f1_macro_0)])
                precision_macro_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="v2/precision_macro", simple_value=precision_macro_0)])
                recall_macro_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="v2/recall_macro", simple_value=recall_macro_0)])
                f1_micro_summary = tf.Summary(value=[tf.Summary.Value(tag="v2/f1_micro", simple_value=f1_micro_0)])
                precision_micro_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="v2/precision_micro", simple_value=precision_micro_0)])
                recall_micro_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="v2/recall_micro", simple_value=recall_micro_0)])

                writer.add_summary(acc_summary, epoch)
                writer.add_summary(nmi_summary, epoch)
                writer.add_summary(adjscore_summary, epoch)
                writer.add_summary(f1_macro, epoch)
                writer.add_summary(precision_macro_summary, epoch)
                writer.add_summary(recall_macro_summary, epoch)
                writer.add_summary(f1_micro_summary, epoch)
                writer.add_summary(precision_micro_summary, epoch)
                writer.add_summary(recall_micro_summary, epoch)

        np.savetxt("./embedding.csv", emb, delimiter="  ")
        print("save embedding finished")
        np.savetxt("./labels.csv", feas['true_labels'], delimiter=",")
        print("save lables finished")