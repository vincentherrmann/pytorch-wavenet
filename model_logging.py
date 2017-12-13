import tensorflow as tf
import numpy as np
import scipy.misc
import threading

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger:
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.generate_interval = generate_interval
        self.accumulated_loss = 0
        self.generate_function = generate_function
        if self.generate_function is not None:
            self.generate_thread = threading.Thread(target=self.generate_function)
            self.generate_function.daemon = True

    def log(self, current_step, current_loss):
        self.accumulated_loss += current_loss
        if current_step % self.log_interval == 0:
            self.log_loss(current_step)
            self.accumulated_loss = 0
        if current_step % self.validation_interval == 0:
            self.validate(current_step)
        if current_step % self.generate_interval == 0:
            self.generate(current_step)

    def log_loss(self, current_step):
        avg_loss = self.accumulated_loss / self.log_interval
        print("loss at step " + str(current_step) + ": " + str(avg_loss))

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        print("validation loss: " + str(avg_loss))
        print("validation accuracy: " + str(avg_accuracy * 100) + "%")

    def generate(self, current_step):
        if self.generate_function is None:
            return

        if self.generate_thread.is_alive():
            print("Last generate is still running, skipping this one")
        else:
            self.generate_thread = threading.Thread(target=self.generate_function,
                                                    args=[current_step])
            self.generate_thread.daemon = True
            self.generate_thread.start()


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class TensorboardLogger(Logger):
    def __init__(self,
                 log_interval=50,
                 validation_interval=200,
                 generate_interval=500,
                 trainer=None,
                 generate_function=None,
                 log_dir='logs'):
        super().__init__(log_interval, validation_interval, generate_interval, trainer, generate_function)
        self.writer = tf.summary.FileWriter(log_dir)

    def log_loss(self, current_step):
        # loss
        avg_loss = self.accumulated_loss / self.log_interval
        self.scalar_summary('loss', avg_loss, current_step)

        # parameter histograms
        for tag, value, in self.trainer.model.named_parameters():
            tag = tag.replace('.', '/')
            self.histo_summary(tag, value.data.cpu().numpy(), current_step)
            if value.grad is not None:
                self.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), current_step)

    def validate(self, current_step):
        avg_loss, avg_accuracy = self.trainer.validate()
        self.scalar_summary('validation loss', avg_loss, current_step)
        self.scalar_summary('validation accuracy', avg_accuracy, current_step)

    def log_audio(self, step):
        samples = self.generate_function()
        tf_samples = tf.convert_to_tensor(samples)
        self.audio_summary('audio sample', tf_samples, step, sr=16000)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def audio_summary(self, tag, sample, step, sr=16000):
        with tf.Session() as sess:
            audio_summary = tf.summary.audio(tag, sample, sample_rate=sr, max_outputs=4)
            summary = sess.run(audio_summary)
            self.writer.add_summary(summary, step)
            self.writer.flush()


    def histo_summary(self, tag, values, step, bins=200):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def tensor_summary(self, tag, tensor, step):
        tf_tensor = tf.Variable(tensor).to_proto()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, tensor=tf_tensor)])
        #summary = tf.summary.tensor_summary(name=tag, tensor=tensor)
        self.writer.add_summary(summary, step)

