import tensorflow as tf

def mse_loss(data_batch, model_pred):
	loss = tf.math.reduce_mean(tf.square(data_batch['label'] - model_pred))
	return loss

def cross_ent_loss(data_batch, model_pred):
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
	    labels=tf.squeeze(data_batch['label']),
	    logits=model_pred,
	)
	return loss