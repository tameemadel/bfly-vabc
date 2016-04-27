import numpy as np
import pylab as pp
import scipy as sp
from scipy import special
import tensorflow as tf
from exponential import *

def logweibull( x, lam, k ):
  log_pdf = np.log(k) - k*np.log(lam) + (k-1.0)*np.log(x) - pow( x / lam, k )
  return log_pdf
  
epsilon = 0.25
n = 1
M = 50
nbr_parameters = 1
S = 10

def Q_0_nu( S ):
  return np.random.rand( S,1 ).astype('float32')

def Q_0_w( S, M ):
  return np.random.rand( S, M ).astype('float32')
  return np.random.rand( 2, S, M ).astype('float32')
  
sy = 5.0
prior_alpha = 2.0
prior_beta  = 0.5

mean_prior  = prior_alpha/prior_beta
var_prior   = prior_alpha/prior_beta**2

phi_init_mean = np.log(mean_prior)
phi_init_log_std = np.log(np.sqrt(var_prior))

p = ExponentialProblem( M, sy, prior_alpha, prior_beta, np.linspace(0.001,1.0,200) )
post_alpha = prior_alpha + M
post_beta  = prior_beta + M*sy

mean_post = post_alpha/post_beta
var_post   = post_alpha/post_beta**2
phi_post_mean = np.log(mean_post)
phi_post_log_std = np.log(np.sqrt(var_post))
#sx_ = tf.placeholder("float", shape=[1], name="sx_")
#sy_ = tf.placeholder("float", shape=[None, 1], name="sy_")

# Q_phi(theta): variational distribution for posterior of theta
# phi[0] -> mean of normal, phi[1] -> log of stddev of normal
phi = tf.Variable(0*tf.ones([2]))

q_log_alpha_uniform = tf.Variable(0*tf.ones([1]))
q_log_beta_uniform  = tf.Variable(0*tf.ones([1]))

q_weibull_log_scale_lambda = tf.Variable(1.25 + 0*tf.ones([1]))
q_weibull_log_shape_k = tf.Variable(2*tf.ones([1]))

q_weibull_scale_lambda = tf.exp( q_weibull_log_scale_lambda )
q_weibull_shape_k = tf.exp( q_weibull_log_shape_k )


# place holders from samples from Q_0
nu = tf.placeholder("float", shape=[S,1], name="nu")
w = tf.placeholder("float", shape=[S,M], name="w")

# see https://en.wikipedia.org/wiki/Weibull_distribution
g1_theta = q_weibull_scale_lambda*tf.pow(-tf.log(nu), 1.0/q_weibull_shape_k)
g1_log_theta = tf.log(g1_theta)
# forward simulation, using plug-ins for u and theta
#x = - tf.log( 1.0 - g2_u ) / g1_theta 
x = - tf.log( 1.0 - w ) / g1_theta 
# statistics of forward simulation
sx = tf.reduce_mean( x, 1 )
vx = tf.reduce_mean( x*x, 1 ) - sx*sx
stdx = tf.sqrt(vx)

# prior for theta is Gamma( alpha, beta )

#alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
log_prior_theta = prior_alpha*np.log(prior_beta) + (prior_alpha-1.0)*g1_log_theta  - prior_beta*g1_theta - special.gammaln(prior_alpha)

# prior for u is uniform(0,1)
log_prior_u = 0.0

# log Q_phi = log normal (g1_theta | phi[0],exp(phi[1])**2)
#log_q_phi = -0.5*np.sqrt(2*np.pi)-g1_log_theta-phi[1]-0.5*tf.pow(g1_log_theta-phi[0],2.0)/tf.pow(tf.exp(phi[1]),2.0)

log_const = q_weibull_shape_k*(g1_log_theta - q_weibull_log_scale_lambda)
log_q_phi = q_weibull_log_shape_k - q_weibull_shape_k*q_weibull_log_scale_lambda + (q_weibull_shape_k-1.0)*g1_log_theta - tf.exp(log_const)

#log_stirling_beta_function = 0.5*np.log(2*np.pi) + (alpha_g2-0.5)*tf.log(alpha_g2) + (beta_g2-0.5)*tf.log(beta_g2) \
#                             -(alpha_g2+beta_g2-0.5)*tf.log(alpha_g2+beta_g2)
# log Q_xi = log beta( g2_u | alpha_g2, beta_g2 ) 
#log_q_xi = (alpha_g2-1.0)*tf.log(g2_u) + (beta_g2-1.0)*tf.log(1.0-g2_u) - log_stirling_beta_function

eps_x = stdx/np.sqrt(M)
# finalize objective function
#log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-np.log(epsilon)-0.5*tf.pow(sy - sx,2.0)/epsilon**2 )
log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x)-0.5*tf.pow(sy - sx,2.0)/tf.pow(eps_x,2) )
#log_likelihood = M*tf.reduce_mean( g1_log_theta - g1_theta*sy )
log_prior = tf.reduce_mean( log_prior_theta ) # + tf.reduce_mean( log_prior_u )
log_q     = tf.reduce_mean( log_q_phi ) #+ tf.reduce_mean( log_q_xi )

# variational lower bound
vlb = log_likelihood + log_prior - log_q
#vlb =  log_prior - log_q
objective_function = -vlb 

sess = tf.InteractiveSession()

print "========================================"
print "1- check the prior is correct, why so off from true posterior? "
print "2- kl divergence with true posterior"
print "3- thetas from Q_phi(theta) have high prob under prior"
print "4- mistakes converting gamma alpha/beta to lognormal means/vars"
print "5- truncated normal instead of lognormal"
print "6- variational for long tailed posterior, gamma?"
print "7- non-rejection based generator for gamma? -- see wikipedia"
print "========================================"



#train_step = tf.train.GradientDescentOptimizer(5*1e-4).minimize(objective_function)
#train_step = tf.train.AdamOptimizer(5*1e-4).minimize(objective_function)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(objective_function)
train_step = tf.train.AdamOptimizer(5*1e-2).minimize(objective_function)
#train_step = tf.train.AdagradOptimizer(1e-4).minimize(objective_function)

sess.run(tf.initialize_all_variables())

last_sx = np.zeros(1)
O = []
for i in range(1500):
  #batch = mnist.train.next_batch(100)
  this_nu = Q_0_nu( S )
  this_w = Q_0_w( S, M ) 
  #this_sx = np.array()
  
  train_step.run(feed_dict={nu: this_nu, w:this_w } )
  #train_step.run(feed_dict={nu: this_nu, w: this_w } ) #, sx_: np.array([sx.eval()]) })
  #last_sx = np.array( [sx.eval()])
  if i%10 == 0:
    this_scale_lambda = q_weibull_scale_lambda.eval()[0]
    this_shape_k = q_weibull_shape_k.eval()[0]
    
    #print log_q.eval(feed_dict={nu: this_nu, w:this_w })
    this_sx = sx.eval(feed_dict={nu: this_nu, w:this_w })
    this_log_prior = log_prior.eval(feed_dict={nu: this_nu, w:this_w })
    this_log_q = log_q.eval(feed_dict={nu: this_nu, w:this_w })
    obj_func = objective_function.eval(feed_dict={nu: this_nu, w:this_w })
    print "tensorflow -- step %d, objective_function %g, mean_x %g +- %g  mean_y %g  log_prior %g  log_q %g  lambda %g k %g"%(i, \
         obj_func , this_sx.mean(), this_sx.std(), sy,this_log_prior, this_log_q,  this_scale_lambda, this_shape_k )
    O.append(obj_func)
  # if i%100 == 0:
  #   #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  #   #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  #   #train_accuracy = accuracy.eval(feed_dict={
  #   #    x:batch[0], y_: batch[1], keep_prob: 1.0})
  #   print "step %d, objective_function %g"%(i, objective_function.eval() )
  #   #print "sum( W**2 ) = %g"%(tf.reduce_sum( tf.pow(W,2) ).eval())
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  
pp.close('all')
pp.figure(1)
#pp.clf()
pp.subplot(1,2,1)
pp.plot(O)
pp.title("TensorFlow")

#mean_post_guess = np.exp(this_phi[0]+0.5*np.exp(this_phi[1])**2)
#var_post_guess = np.exp( np.exp(this_phi[1])**2 - 1.0)*np.exp(2*this_phi[0]+np.exp(this_phi[1])**2)

# a/b = m, a = bm, v = a/b*b, v = m / b, b = m/v, a = m*m/v
#post_beta_guess = mean_post_guess/var_post_guess
#post_alpha_guess = mean_post_guess*mean_post_guess/var_post_guess
#M, sy, alpha, beta, theta_grid 
#p2 = ExponentialProblem( 0, 0, post_alpha_guess, post_beta_guess, p.fine_theta_range )
log_model = logweibull( p.fine_theta_range, this_scale_lambda, this_shape_k )
#log_model=lognormal_logpdf(p.fine_theta_range, this_phi[0], np.exp(this_phi[1]))
#lm = spstats.lognorm( np.exp(this_phi[1]),0,this_phi[0]).pdf( p.fine_theta_range )
pp.subplot(1,2,2)
pp.plot( p.fine_theta_range, p.posterior, lw=2)
#pp.plot( p.fine_theta_range, p2.posterior, lw=2)
pp.plot( p.fine_theta_range, np.exp(log_model), lw=2)
#pp.plot( p.fine_theta_range, np.exp(log_model), lw=2)
pp.legend( ["true", "vb"])
pp.show()
