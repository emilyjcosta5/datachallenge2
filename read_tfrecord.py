import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

def read_tfrecord(filename):
    
    tfdataset = tf.data.TFRecordDataset([filename])
    def _parse_function(example_proto):
        # Create a description of the features.
        feature_description = {
            'image_1': tf.io.FixedLenFeature([], dtype=tf.string),
            'image_2': tf.io.FixedLenFeature([], dtype=tf.string),
            'image_3': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)

        def reshape_image(image_name):
            image = tf.decode_raw(parsed[image_name], tf.int64)
            height = tf.cast(512, tf.int32)
            width = tf.cast(512, tf.int32)
            image = tf.reshape(image, [height, width])
            return image
        
        image_1 = reshape_image('image_1')
        image_2 = reshape_image('image_2')
        image_3 = reshape_image('image_3')
        label = tf.cast(parsed['label'], tf.string)
        #images = {'image_1': image_1, 'image_2': image_2, 'image_3': image_3}
        images = tf.stack([image_1, image_2, image_3])
        print(images)

        #onehot_encoder = OneHotEncoder(sparse=False, n_values=230)
        #label = onehot_encoder.fit_transform(label)

        return images, label

    parsed_dataset = tfdataset.map(_parse_function)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)

    return iterator.get_next()
    

if __name__ == '__main__':
    h5_path = "/ccs/home/shutoaraki/challenge_all_data/train"
    filename = os.path.join(h5_path, 'train-00000-of-01024.tfrecords')
    print(filename)
    images, label = read_tfrecord(filename)
    

    print(label)
    print(images)
