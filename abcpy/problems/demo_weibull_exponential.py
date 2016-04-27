import sys;
sys.path.append('/home/tameem/.spyder2')

import numpy as np
import pylab as pp
import scipy as sp
from scipy import special
import tensorflow as tf
from abcpy.problems.exponential import *
from abcpy.plotting import *

"""
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")
# Create an Op to add one to `state`.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))
"""

import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = StringIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))

def logweibull( x, lam, k ):
  log_pdf = np.log(k) - k*np.log(lam) + (k-1.0)*np.log(x) - pow( x / lam, k )
  return log_pdf
  
def logofGaussian( x, q_mu, q_sigma ):
  log_pdf = -np.log(q_sigma * np.power(2 * np.pi, 0.5)) -  (np.power(x - q_mu,2) / (2.0 * np.power(q_sigma,2)))
  return log_pdf
  
#def lognormal( x, mu, sigma_2 ):
  #log_pdf = (() / (x ())) np.log(k) - k*np.log(lam) + (k-1.0)*np.log(x) - pow( x / lam, k )
  
   #-log(pr_sigma_log_P * np.sqrt(2 * np.pi)) -  ((g1_theta01 - pr_mu_log_P)**2 / (2 * pr_sigma_log_P**2))
   
  return log_pdf

def default_params():
  params = {}
  params["alpha"]           = 0.1
  params["beta"]            = 0.1
  params["theta_star"]      = 0.1
  params["N"]               = 500  # how many observations we draw per simulation
  params["q_stddev"]        = 0.5
  params["epsilon"]         = 0.1
  params["use_model"]       = False
  return params
  
sess = tf.InteractiveSession()

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
  
# Initial Conditions -- some rain drops hit a pond


epsilon = 0.25
n = 1
M = 50
nbr_parameters = 5
S = 1

#def Q_0_nu( S ):
  ##return np.random.rand( S,1 ).astype('float32')
  #return np.random.rand( S, nbr_parameters ).astype('float32')
  
def Q_0_nu( S, nbr_parameters ):
  #return np.random.rand( S,1 ).astype('float32')
  return np.random.rand( S, nbr_parameters ).astype('float32')

def Q_0_w( S, M ):
  return np.random.rand( S, M ).astype('float32')
  return np.random.rand( 2, S, M ).astype('float32')
  
def calc_synth_data( w, g1_th01, g1_th02, g1_th03, g1_th04, g1_th05, tau, index):
    
    #sess = tf.InteractiveSession()  
    #sess.run(tf.initialize_all_variables())
    
    #log_P       = g1_th01
    #log_delta   = g1_th02
    #log_N0      = g1_th03
    #log_sigma_d = g1_th04
    #log_sigma_p = g1_th05    
    #N0, P, sigma_p, sigma_d, delta = tf.exp(log_N0), tf.exp(log_P), tf.exp(log_sigma_p), tf.exp(log_sigma_d), tf.exp(log_delta)
    P       = g1_th01
    delta   = g1_th02
    N0      = g1_th03
    sigma_d = g1_th04
    sigma_p = g1_th05
    
    T = 180
    #var_d  = sigma_d**2
    var_d = tf.pow(sigma_d, 2)
    prec_d = 1.0 / var_d
    
    #var_p  = sigma_p**2
    var_p = tf.pow(sigma_p, 2)
    prec_p = 1.0 / var_p
    
    burnin = 50
    lag = int(np.floor(tau))
    if (float(tau)-float(lag)>0.5):
      lag = lag + 1

    #print N0
    #N[0] = N0
    #N = tf.Variable(tf.zeros([lag+burnin+T, 1], tf.float32))
    #N = tf.Variable(tf.zeros([T+1, 1], tf.float32))
    N_temp = tf.Variable(tf.zeros([T+1, 1], tf.float32))
    N_temp2 = tf.Variable(tf.zeros([lag+burnin+T, 1], tf.float32))
    N_temp = np.zeros([lag+burnin+T, 1])
    N_temp2 = np.zeros(T+1)
    
    #q_mu_log_P
    sess16 = tf.InteractiveSession()
    sess16.run(tf.initialize_all_variables())
    this_nu = Q_0_nu( S, nbr_parameters )
    this_w = Q_0_w( S, M )
    N_temp[0] = N0.eval(feed_dict={nu: this_nu, w:this_w })
    
    for i in range(lag):
        N_temp[i] = 180.0
        
    for i in xrange(burnin+T):
        t = i + lag
        
        #eps_t = gamma_rnd( prec_d.eval(feed_dict={nu: this_nu, w:this_w }), prec_d.eval(feed_dict={nu: this_nu, w:this_w }) )
        #e_t   = gamma_rnd( prec_p.eval(feed_dict={nu: this_nu, w:this_w }), prec_p.eval(feed_dict={nu: this_nu, w:this_w }) )
        log_eps_t = tf.random_normal([1], -tf.log(var_d + 1)/2 , tf.log(var_d + 1))
        eps_t = tf.exp(log_eps_t)
        log_e_t = tf.random_normal([1], -tf.log(var_p + 1)/2 , tf.log(var_p + 1))
        e_t = tf.exp(log_e_t)
        
        tau_t = t - lag
        N_temp[t] = P.eval(feed_dict={nu: this_nu, w:this_w })*N_temp[tau_t]*\
        np.exp(-N_temp[tau_t]/N0.eval(feed_dict={nu: this_nu, w:this_w }))*e_t.eval(feed_dict={nu: this_nu, w:this_w })\
        + N_temp[t-1]*np.exp(-delta.eval(feed_dict={nu: this_nu, w:this_w })*eps_t.eval(feed_dict={nu: this_nu, w:this_w }))
    
    N_temp2 = N_temp[-(T+1):]
    #N = tf.assign(N, N_temp2)
    N = tf.Variable(N_temp2)
    return N_temp2
    #N_summ = tf.scalar_summary("N", N)    
    #merged = tf.merge_all_summaries()
    
    #with tf.Session() as sess5:    
    #    writer = tf.train.SummaryWriter("./", sess5.graph_def)
    #    tf.initialize_all_variables().run()
    #    this_nu = Q_0_nu( S, nbr_parameters )
    #    this_w = Q_0_w( S, M )
    #    feed = {nu: this_nu, w:this_w }
    #    result = sess5.run(merged, feed_dict=feed)
    #    summary_str = result
    #    writer.add_summary(summary_str, 0)
    #    writer.flush()
    
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess5:
        this_nu = Q_0_nu( S, nbr_parameters )
        this_w = Q_0_w( S, M )
        N_temp[0] = N0.eval(feed_dict={nu: this_nu, w:this_w })
    
        for i in range(lag):
            N_temp[i] = 180.0
            
        for i in xrange(burnin+T):
            t = i + lag
        
            eps_t = gamma_rnd( prec_d.eval(feed_dict={nu: this_nu, w:this_w }), prec_d.eval(feed_dict={nu: this_nu, w:this_w }) )
            e_t   = gamma_rnd( prec_p.eval(feed_dict={nu: this_nu, w:this_w }), prec_p.eval(feed_dict={nu: this_nu, w:this_w }) )
        
            #tau_t = max(0,t-int(tau))
            tau_t = t - lag
            N_temp[t] = P.eval(feed_dict={nu: this_nu, w:this_w })*N_temp[tau_t]*\
            np.exp(-N_temp[tau_t]/N0.eval(feed_dict={nu: this_nu, w:this_w }))*e_t\
            + N_temp[t-1]*np.exp(-delta.eval(feed_dict={nu: this_nu, w:this_w })*eps_t)
        
        N_temp2 = N_temp[-(T+1):]
        N = tf.assign(N, N_temp2)
        #return N
        return N_temp2
        

    #X1 = tf.Variable(tf.zeros([10, 1], tf.float32))
    ####X1 = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=np.float32)
    #X3 = tf.Variable(tf.zeros([10, 1], tf.float32)); 
    #X3 = np.zeros([10, 1])
    ####elec = tf.Variable(44.33)
    ####wise = tf.Variable(44.33)
    #X1 = tf.assign(X1, X3)
    #X3[0,0] = 65
    #X1 = tf.assign(X1, X3)
    
    ###ff = tf.Variable(22); gg = tf.Variable(11); ff = tf.assign(ff, gg)
    ###ss = tf.Variable(tf.ones([10])); ff = tf.ass

#sy = 5.0
blowfly_filename = "/home/tameem/.spyder2/abcpy/abcpy/problems/blowfly/blowfly.txt"
observations   = np.loadtxt( blowfly_filename )[:,1]
sy = observations.mean()


prior_alpha = 2.0
prior_beta  = 0.5

mean_prior  = prior_alpha/prior_beta
var_prior   = prior_alpha/prior_beta**2

phi_init_mean = np.log(mean_prior)
phi_init_log_std = np.log(np.sqrt(var_prior))

params = {}
params["M"] = M
params["sy"] = sy
params["prior_alpha"] = prior_alpha
params["prior_beta"] = prior_beta  # how many observations we draw per simulation
params["fine_range"] = np.linspace(0.001,1.0,200)
#params["epsilon"] = 0.1
#params["use_model"] = False

#p = ExponentialProblem( M, sy, prior_alpha, prior_beta, np.linspace(0.001,1.0,200) )
alpha = prior_alpha; beta = prior_beta;
p = ExponentialProblem(alpha, beta)
#p = ExponentialProblem( params )
post_alpha = prior_alpha + M
post_beta  = prior_beta + M*sy

mean_post = post_alpha/post_beta
var_post   = post_alpha/post_beta**2
phi_post_mean = np.log(mean_post)
phi_post_log_std = np.log(np.sqrt(var_post))
#sx_ = tf.placeholder("float", shape=[1], name="sx_")
#sy_ = tf.placeholder("float", shape=[None, 1], name="sy_")

with tf.Session() as sess76:
    count = tf.Variable(0)
    updater = tf.assign(count, count + 1)
    # Q_phi(theta): variational distribution for posterior of theta
    # phi[0] -> mean of normal, phi[1] -> log of stddev of normal
    phi = tf.Variable(0*tf.ones([2]), name="phi")

    q_log_alpha_uniform = tf.Variable(0*tf.ones([1]), name="q_log_alpha_uniform")
    q_log_beta_uniform  = tf.Variable(0*tf.ones([1]), name="q_log_beta_uniform")
    
    q_weibull_log_scale_lambda = tf.Variable(1.25 + 0*tf.ones([1]), name="q_weibull_log_scale_lambda")
    q_weibull_log_shape_k = tf.Variable(2*tf.ones([1]), name="q_weibull_log_shape_k")
    #nbr_parameters = 6
    mu_log_P         = 2.0
    std_log_P        = 2.0
    mu_log_delta     = -1.0
    std_log_delta    = 2.0
    mu_log_N0        = 5.0
    std_log_N0       = 2.0
    mu_log_sigma_d   = 0.0
    std_log_sigma_d  = 2.0
    mu_log_sigma_p   = 0.0
    std_log_sigma_p  = 2.0
    mu_tau           = 15
    q_factor         = 0.1
    epsilon          = 10
    
    q_mu_log_P = tf.Variable(2.0, name="q_mu_log_P")
    q_log_sigma_log_P = tf.Variable(0.6913, name="q_log_sigma_log_P")
    #q_sigma_log_P = tf.Variable(0.0, name="q_sigma_log_P")
    #q_sigma_log_P = tf.exp(q_log_sigma_log_P)
    q_sigma_log_P = tf.exp(0.6913)
    
    q_mu_log_delta = tf.Variable(-1.0, name="q_mu_log_delta")
    q_log_sigma_log_delta = tf.Variable(0.6913, name="q_log_sigma_log_delta")
    #q_sigma_log_delta = tf.exp(q_log_sigma_log_delta)
    q_sigma_log_delta = tf.exp(0.6913)

    q_mu_log_N0 = tf.Variable(5.0, name="q_mu_log_N0")
    q_log_sigma_log_N0 = tf.Variable(0.6913, name="q_log_sigma_log_N0")
    #q_sigma_log_N0 = tf.Variable(0.0, name="q_sigma_log_N0")
    #q_sigma_log_N0 = tf.exp(q_log_sigma_log_N0)
    q_sigma_log_N0 = tf.exp(0.6913)
    
    q_mu_log_sigma_d = tf.Variable(0.0, name="q_mu_log_sigma_d")
    q_log_sigma_log_sigma_d = tf.Variable(0.6913, name="q_log_sigma_log_sigma_d")
    #q_sigma_log_sigma_d = tf.Variable(0.0, name="q_sigma_log_sigma_d")
    #q_sigma_log_sigma_d = tf.exp(q_log_sigma_log_sigma_d)
    q_sigma_log_sigma_d = tf.exp(0.6913)

    q_mu_log_sigma_p = tf.Variable(0.0, name="q_mu_log_sigma_p")
    q_log_sigma_log_sigma_p = tf.Variable(0.6913, name="q_log_sigma_log_sigma_p")
    #q_sigma_log_sigma_p = tf.Variable(0.0, name="q_sigma_log_sigma_p")
    #q_sigma_log_sigma_p = tf.exp(q_log_sigma_log_sigma_p)
    q_sigma_log_sigma_p = tf.exp(0.6913)
    
    pr_mu_log_P         = 2.0
    pr_sigma_log_P        = 2.0
    pr_mu_log_delta     = -1.0
    pr_sigma_log_delta    = 2.0
    pr_mu_log_N0        = 5.0
    pr_sigma_log_N0       = 2.0
    pr_mu_log_sigma_d   = 0.0
    pr_sigma_log_sigma_d  = 2.0
    pr_mu_log_sigma_p   = 0.0
    pr_sigma_log_sigma_p  = 2.0
    mu_tau           = 15
    q_factor         = 0.1
    epsilon          = 10
    
    q_weibull_scale_lambda = tf.exp( q_weibull_log_scale_lambda )
    q_weibull_shape_k = tf.exp( q_weibull_log_shape_k )
    
    # place holders from samples from Q_0
    nu = tf.placeholder("float", shape=[S, nbr_parameters], name="nu")
    w = tf.placeholder("float", shape=[S, M], name="w")
    #nu = tf.placeholder("float", shape=[S,nbr_parameters], name="nu")
    #w = tf.placeholder("float", shape=[S,M], name="w")
    
    #ggg = tf.Variable(1)
    #ggg = tf.Variable(tf.zeros([2]))
    #hh = [3, 4, 55]
    #g1_theta = tf.Variable(tf.zeros([S, nbr_parameters]))
    g1_theta = tf.placeholder("float", shape=[S, nbr_parameters], name="g1_theta")
    #g1_theta_summ = tf.histogram_summary("g1_theta", g1_theta)
    #Weibull:
    #g1_theta = [mu_log_P*tf.pow(-tf.log(nu[:,0]),1.0/std_log_P),\
    #mu_log_delta*tf.pow(-tf.log(nu[:,1]),1.0/std_log_delta),mu_log_N0*tf.pow(-tf.log(nu[:,2]),1.0/std_log_N0),\
    #mu_log_sigma_d*tf.pow(-tf.log(nu[:,3]),1.0/std_log_sigma_d),mu_log_sigma_p*tf.pow(-tf.log(nu[:,4]),1.0/std_log_sigma_p)]
    #Normal:
    #mu_log_P + std_log_P*np.random.randn(1)[0]
    #g1_theta = [mu_log_P+std_log_P*nu[:,0],\
    #mu_log_delta+std_log_delta*nu[:,1],mu_log_N0+std_log_N0*nu[:,2],\
    #mu_log_sigma_d+std_log_sigma_d*nu[:,3],mu_log_sigma_p+std_log_sigma_p*nu[:,4]]

    standbyme = tf.Variable(tf.zeros([S]), name="standbyme")
    
    ##x = - tf.log( 1.0 - g2_u ) / g1_theta
    ##x = - tf.log( 1.0 - w ) / g1_theta
    #x = - tf.log( 1.0 - w ) / g1_th0
  
    #g1_log_theta01 = tf.Variable(0.0, name="g1_log_theta01")
    #g1_log_theta02 = tf.Variable(0.0, name="g1_log_theta02")
    #g1_log_theta03 = tf.Variable(0.0, name="g1_log_theta03")
    #g1_log_theta04 = tf.Variable(0.0, name="g1_log_theta04")
    #g1_log_theta05 = tf.Variable(0.0, name="g1_log_theta05")
    #g1_theta01 = tf.Variable(0.0, name="g1_theta01")
    #g1_theta02 = tf.Variable(0.0, name="g1_theta02")
    #g1_theta03 = tf.Variable(0.0, name="g1_theta03")
    #g1_theta04 = tf.Variable(0.0, name="g1_theta04")
    #g1_theta05 = tf.Variable(0.0, name="g1_theta05")
    g1_log_theta01 = tf.Variable(tf.zeros([S]), name='g1_log_theta01')
    g1_log_theta02 = tf.Variable(tf.zeros([S]), name='g1_log_theta02')
    g1_log_theta03 = tf.Variable(tf.zeros([S]), name='g1_log_theta03')
    g1_log_theta04 = tf.Variable(tf.zeros([S]), name='g1_log_theta04')
    g1_log_theta05 = tf.Variable(tf.zeros([S]), name='g1_log_theta05')
    #g1_log_theta01 = q_mu_log_P + q_sigma_log_P*nu[:, 0]
    g1_log_theta01 = tf.log(q_mu_log_P*tf.pow(-tf.log(nu[:, 0]), 1.0/q_sigma_log_P))
    g1_log_theta02 = q_mu_log_delta + q_sigma_log_delta*nu[:, 1]
    g1_log_theta03 = q_mu_log_N0 + q_sigma_log_N0*nu[:, 2]
    g1_log_theta04 = q_mu_log_sigma_d + q_sigma_log_sigma_d*nu[:, 3]
    g1_log_theta05 = q_mu_log_sigma_p + q_sigma_log_sigma_p*nu[:, 4]
    g1_theta01 = tf.exp(g1_log_theta01)
    g1_theta02 = tf.exp(g1_log_theta02)
    g1_theta03 = tf.exp(g1_log_theta03)
    g1_theta04 = tf.exp(g1_log_theta04)
    g1_theta05 = tf.exp(g1_log_theta05)
    with tf.Session() as sess335:
        g1_theta01_summ = tf.scalar_summary("g1_theta01", g1_theta01)
        g1_theta02_summ = tf.scalar_summary("g1_theta02", g1_theta02)
        g1_theta03_summ = tf.scalar_summary("g1_theta03", g1_theta03)
        g1_theta04_summ = tf.scalar_summary("g1_theta04", g1_theta04)
        g1_theta05_summ = tf.scalar_summary("g1_theta05", g1_theta05)
        merged = tf.merge_all_summaries()
        #    writer.flush()
        #    sadsa = 44
  
    T = 180
    mini_S = 1
    x = tf.Variable(tf.zeros([mini_S, T+1], tf.float32), name="x")
    #x_temp = tf.Variable(tf.zeros([S, T+1], tf.float32))
    x_temp = np.zeros([mini_S, T+1], dtype=np.float32)
    ##for index in range(S):
    for index in range(mini_S):
        dd = calc_synth_data(w, g1_theta01[index], g1_theta02[index], \
        g1_theta03[index],g1_theta04[index],g1_theta05[index], \
        poisson_rand(mu_tau), index)
        dd = np.reshape(dd, (T+1))
        x_temp[index, :] = dd
    x = tf.Variable(x_temp, name="x")
    sx = tf.reduce_mean( x, 1 )
    vx = tf.reduce_mean( x*x, 1 ) - sx*sx
    stdx = tf.sqrt(vx)
    ##dd = calc_synth_data(w, mu_log_P+std_log_P*nu[index,0], mu_log_delta+std_log_delta*nu[index,1],\
    ##mu_log_N0+std_log_N0*nu[index,2],mu_log_sigma_d+std_log_sigma_d*nu[index,3],\
    ##mu_log_sigma_p+std_log_sigma_p*nu[index,4], poisson_rand(mu_tau), index)
    #dd = np.reshape(dd, (T+1))
    #x_temp[index, :] = dd
    
    
    
    #log_prior_theta_0 = tf.Variable(-tf.log(pr_sigma_log_P) * tf.pow(2 * np.pi, 0.5) - (33 - pr_mu_log_P)**2 )
    #log_prior_theta_1 = tf.Variable(-tf.log(pr_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - pr_mu_log_P,2) / (2 * tf.pow(pr_sigma_log_P,2))), name="log_prior_theta_1")
    #log_prior_theta_1 = tf.Variable(0.0, name="log_prior_theta_1")
    #sss = tf.Variable(g1_theta02, name="sss")
    log_prior_theta_1 = tf.Variable(-tf.log(pr_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - pr_mu_log_P,2) / (2.0 * tf.pow(pr_sigma_log_P,2))))
    log_prior_theta_2 = tf.Variable(-tf.log(pr_sigma_log_delta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta02 - pr_mu_log_delta,2) / (2 * tf.pow(pr_sigma_log_delta,2))))
    log_prior_theta_3 = tf.Variable(-tf.log(pr_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - pr_mu_log_N0,2) / (2 * tf.pow(pr_sigma_log_N0,2))))
    log_prior_theta_4 = tf.Variable(-tf.log(pr_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - pr_mu_log_sigma_d,2) / (2 * tf.pow(pr_sigma_log_sigma_d,2))))
    log_prior_theta_5 = tf.Variable(-tf.log(pr_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - pr_mu_log_sigma_p,2) / (2 * tf.pow(pr_sigma_log_sigma_p,2))))
    ##log_prior_theta_5 = -log(pr_sigma_log_sigma_p * np.sqrt(2 * np.pi)) -  ((g1_theta05 - pr_mu_log_sigma_p)**2 / (2 * pr_sigma_log_sigma_p**2))
        
    # prior for u is uniform(0,1)
    log_prior_u = 0.0
        
    log_q_theta_1 = tf.Variable(-tf.log(q_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - q_mu_log_P,2) / (2 * tf.pow(q_sigma_log_P,2))))
    log_q_theta_2 = tf.Variable(-tf.log(q_sigma_log_delta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta02 - q_mu_log_delta,2) / (2 * tf.pow(q_sigma_log_delta,2))))
    log_q_theta_3 = tf.Variable(-tf.log(q_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - q_mu_log_N0,2) / (2 * tf.pow(q_sigma_log_N0,2))))
    log_q_theta_4 = tf.Variable(-tf.log(q_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - q_mu_log_sigma_d,2) / (2 * tf.pow(q_sigma_log_sigma_d,2))))
    log_q_theta_5 = tf.Variable(-tf.log(q_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - q_mu_log_sigma_p,2) / (2 * tf.pow(q_sigma_log_sigma_p,2))))    
    
    eps_x = stdx/np.sqrt(M)
    # finalize objective function
    #log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-np.log(epsilon)-0.5*tf.pow(sy - sx,2.0)/epsilon**2 )
    log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x)-0.5*tf.pow(sy - sx,2.0)/tf.pow(eps_x,2) )
    #log_likelihood = tf.cast(log_likelihood, tf.float32)
    #log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x)-0.5*tf.pow(sy - sx,2.0)/tf.pow(eps_x,2) )
    ##log_likelihood = M*tf.reduce_mean( g1_log_theta - g1_theta*sy )
    #log_prior = tf.reduce_mean( log_prior_theta ) # + tf.reduce_mean( log_prior_u )
    #log_q     = tf.reduce_mean( log_q_phi ) #+ tf.reduce_mean( log_q_xi )
    log_prior_1 = tf.reduce_mean( log_prior_theta_1 )
    log_prior_2 = tf.reduce_mean( log_prior_theta_2 )
    log_prior_3 = tf.reduce_mean( log_prior_theta_3 )
    log_prior_4 = tf.reduce_mean( log_prior_theta_4 )
    log_prior_5 = tf.reduce_mean( log_prior_theta_5 )
    log_q_1     = tf.reduce_mean( log_q_theta_1 )
    log_q_2     = tf.reduce_mean( log_q_theta_2 )
    log_q_3     = tf.reduce_mean( log_q_theta_3 )
    log_q_4     = tf.reduce_mean( log_q_theta_4 )
    log_q_5     = tf.reduce_mean( log_q_theta_5 )

    # variational lower bound
    #vlb = log_likelihood + log_prior_1 + log_prior_2 + log_prior_3 + log_prior_4 + log_prior_5 - log_q_1 - log_q_2 - log_q_3 - log_q_4 - log_q_5
    tt = tf.Variable(22.1)
    #vlb = tt + log_prior_1 - log_q_1
    vlb = log_likelihood + log_prior_1 - log_q_1
    #vlb = log_likelihood - log_q_1
    #vlb =  log_prior - log_q
    objective_function = -vlb
    
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

    this_nu = Q_0_nu( S, nbr_parameters )
    this_w = Q_0_w( S, M )    
    sess76.run(tf.initialize_all_variables(), feed_dict={nu: this_nu, w:this_w })
    #sess76.run(tf.initialize_all_variables())
    
    #for i in range(1500):
    for i in range(500):
        #batch = mnist.train.next_batch(100)
        this_nu = Q_0_nu( S, nbr_parameters )
        this_w = Q_0_w( S, M )
        sess76.run(train_step, feed_dict={nu: this_nu, w:this_w })

        #train_step.run(feed_dict={nu: this_nu, w:this_w } )
        if i%10 == 0:
            #this_scale_lambda = q_weibull_scale_lambda.eval()[0]
            #this_shape_k = q_weibull_shape_k.eval()[0]
            this_mu_log_P = q_mu_log_P.eval()
            this_sigma_log_P = q_sigma_log_P.eval()
            
            #print log_q.eval(feed_dict={nu: this_nu, w:this_w })
            this_sx = sx.eval(feed_dict={nu: this_nu, w:this_w })
            this_log_prior_1 = log_prior_1.eval(feed_dict={nu: this_nu, w:this_w })
            this_log_q_1 = log_q_1.eval(feed_dict={nu: this_nu, w:this_w })
            obj_func = objective_function.eval(feed_dict={nu: this_nu, w:this_w })
            #print "tensorflow -- step %d, objective_function %g, mean_x %g +- %g  mean_y %g  log_prior %g  log_q %g  lambda %g k %g"%(i, \
            #obj_func , this_sx.mean(), this_sx.std(), sy,this_log_prior_1, this_log_q_1,  this_scale_lambda, this_shape_k )
            print "tensorflow -- step %d, objective_function %g, mean_x %g +- %g  mean_y %g  log_prior %g  log_q %g mu_p %g sigma_p %g"%(i, \
            obj_func , this_sx.mean(), this_sx.std(), sy,this_log_prior_1, this_log_q_1, this_mu_log_P, this_sigma_log_P)
            dfd = 5
            #O.append(obj_func)
        
        #sess76.run(updater)
        #print("The count is {}".format(sess76.run(count)))
        #this_nu = Q_0_nu( S, nbr_parameters )
        #this_w = Q_0_w( S, M )
        #sess76.run(g1_theta01, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(g1_theta02, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(g1_theta03, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(g1_theta04, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(g1_theta05, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(x, feed_dict={nu: this_nu, w:this_w })
        ###sess76.run(log_prior_theta_1, feed_dict={nu: this_nu, w:this_w })
        ###sess76.run(log_prior_theta_5, feed_dict={nu: this_nu, w:this_w })
        ###sess76.run(log_q_theta_1, feed_dict={nu: this_nu, w:this_w })
        ###sess76.run(log_qr_theta_5, feed_dict={nu: this_nu, w:this_w })
        #sess76.run(phi)
    enough = 7
dummy = 42


min_range = this_mu_log_P - 2 * this_sigma_log_P
max_range = this_mu_log_P + 2 * this_sigma_log_P
#min_range = 0.001
#max_range = 1.0
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title("TensorFlow")
#log_model = logweibull( p.fine_theta_range, this_scale_lambda, this_shape_k )
#dd = np.exp(logofGaussian(0, 0, 1.0))
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_P, this_sigma_log_P)
#log_model_2 = logofGaussian(fine_theta_range, q_mu_log_delta, q_sigma_log_delta)
#log_model_3 = logofGaussian(fine_theta_range, q_mu_log_N0, q_sigma_log_N0)
#log_model_4 = logofGaussian(fine_theta_range, q_mu_log_sigma_d, q_sigma_log_sigma_d)
#log_model_5 = logofGaussian(fine_theta_range, q_mu_log_sigma_p, q_sigma_log_sigma_p)

#pp.subplot(1,2,2)
#pp.plot( p.fine_theta_range, p.posterior, lw=2)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["true", "vb"])
pp.show()
enough = 1

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
#pp.subplot(1,2,2)
#pp.plot( p.fine_theta_range, p.posterior, lw=2)
#pp.plot( p.fine_theta_range, p2.posterior, lw=2)
pp.plot( p.fine_theta_range, np.exp(log_model), lw=2)
#pp.plot( p.fine_theta_range, np.exp(log_model), lw=2)
pp.legend( ["true", "vb"])
pp.show()

#q_mu_log_P_summ = tf.scalar_summary("q_mu_log_P", q_mu_log_P)
#q_sigma_log_P_summ = tf.scalar_summary("q_sigma_log_P", q_sigma_log_P)
#q_mu_log_delta_summ = tf.scalar_summary("q_mu_log_delta", q_mu_log_delta)
#q_sigma_log_delta_summ = tf.scalar_summary("q_sigma_log_delta", q_sigma_log_delta)
#q_mu_log_N0_summ = tf.scalar_summary("q_mu_log_N0", q_mu_log_N0)
#q_sigma_log_N0_summ = tf.scalar_summary("q_sigma_log_N0", q_sigma_log_N0)
#q_mu_log_sigma_d_summ = tf.scalar_summary("q_mu_log_sigma_d", q_mu_log_sigma_d)
#q_sigma_log_sigma_d_summ = tf.scalar_summary("q_sigma_log_sigma_d", q_sigma_log_sigma_d)
#q_mu_log_sigma_p_summ = tf.scalar_summary("q_mu_log_sigma_p", q_mu_log_sigma_p)
#q_sigma_log_sigma_p_summ = tf.scalar_summary("q_sigma_log_sigma_p", q_sigma_log_sigma_p)

    
#theta = np.zeros((nbr_parameters, 1))
#theta[0,0] = mu_log_P + std_log_P*np.random.randn( S ) # np.log(P)
#theta[1,0] = mu_log_delta   + std_log_delta*np.random.randn( N )# np.log(delta)
#theta[2,0] = mu_log_N0      + std_log_N0*np.random.randn( N ) # np.log(N0)
#theta[3,0] = mu_log_sigma_d + std_log_sigma_d*np.random.randn( N )# np.log(sigma_d)
#theta[4,0] = mu_log_sigma_p + std_log_sigma_p*np.random.randn( N ) # np.log(sigma_p)
#theta[5,0] = poisson_rand( mu_tau ) # tau

#theta_0 = tf.exp( log_theta[0] )
#theta_1 = tf.Variable(tf.exp(log_theta[1]))
#theta_2 = tf.Variable(tf.exp(log_theta[2]))
#theta_3 = tf.Variable(tf.exp(log_theta[3]))
#theta_4 = tf.Variable(tf.exp(log_theta[4]))
#theta_5 = tf.Variable(tf.exp(log_theta[5]))
#theta = tf.Variable([theta_0, theta_1, theta_2, theta_3, theta_4, theta_5])


# see https://en.wikipedia.org/wiki/Weibull_distribution
#g1_theta = q_weibull_scale_lambda*tf.pow(-tf.log(nu), 1.0/q_weibull_shape_k)
#g1_log_theta = tf.log(g1_theta)

#g1_theta_00 = tf.Variable(tf.zeros([4])
#g1_theta_00[0] = tf.assign(4)
#g1_theta_00[1] = 12;
#g1_theta_00[2] = 21;
#g1_theta_00[3] = 55;


#g1_th0 = g1_theta[:][0]



    ## Merge all the summaries and write them out to /tmp/mnist_logs
    ##merged = tf.merge_all_summaries()
    ##writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def) 
    #init_op = tf.initialize_all_variables()
    #with tf.Session() as sess5:
    #    writer = tf.train.SummaryWriter("./", sess5.graph_def)
    #    tf.initialize_all_variables().run()
    #    this_nu = Q_0_nu( S, nbr_parameters )
    #    this_w = Q_0_w( S, M )
    #    feed = {nu: thisq_sigma_log_P_nu, w:this_w }
    #    N_tt = g1_log_theta01.eval(feed_dict={nu: this_nu, w:this_w })
    #    result = sess5.run(merged, feed_dict=feed)
    #    summary_str = result
    #    writer.add_summary(summary_str, 0)##x = - tf.log( 1.0 - g2_u ) / g1_theta
##x = - tf.log( 1.0 - w ) / g1_theta
#x = - tf.log( 1.0 - w ) / g1_th0
    
    #init_op = tf.initialize_all_variables()
    #with tf.Session() as sess5:
    #    this_nu = Q_0_nu( S )
    #    this_w = Q_0_w( S, M )
    
    #q_mu_log_P
    #sess12 = tf.InteractiveSession()
    #sess12.run(tf.initialize_all_variables())

    #remove one, and only one, #:
    #dd = calc_synth_data(w, g1_theta01, g1_theta02,\
    #g1_theta03,g1_theta04,g1_theta05, \
    #poisson_rand(mu_tau), index)
    ##dd = calc_synth_data(w, mu_log_P+std_log_P*nu[index,0], mu_log_delta+std_log_delta*nu[index,1],\
    ##mu_log_N0+std_log_N0*nu[index,2],mu_log_sigma_d+std_log_sigma_d*nu[index,3],\
    ##mu_log_sigma_p+std_log_sigma_p*nu[index,4], poisson_rand(mu_tau), index)
    #dd = np.reshape(dd, (T+1))
    #x_temp[index, :] = dd





sess66 = tf.InteractiveSession()

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

sess66.run(tf.initialize_all_variables())

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


##alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
#log_prior_theta = prior_alpha*np.log(prior_beta) + (prior_alpha-1.0)*g1_log_theta  - prior_beta*g1_theta - special.gammaln(prior_alpha)

#*np.power(-np.log(nu[0,0]), 1.0/std_log_P)
init_op = tf.initialize_all_variables()
with tf.Session() as sess4:
  #sess4.run(ggg)
  #print(sess4.run(ggg))
  print(sess4.run(init_op))
  #print(sess4.run(zz))
  this_nu = Q_0_nu( S )
  this_w = Q_0_w( S, M )
  print (g1_theta[0].eval(feed_dict={nu: this_nu, w:this_w }))
  print (g1_theta[1].eval(feed_dict={nu: this_nu, w:this_w }))
  #sess4.run(feed_dict={nu: this_nu, w:this_w } )

"""
g1_theta_0 = mu_log_P*tf.pow(-tf.log(nu[0,0]), 1.0/std_log_P)
g1_log_theta_0 = tf.log(g1_theta_0)
g1_theta_1 = mu_log_delta*tf.pow(-tf.log(nu[1,0]), 1.0/std_log_delta)
g1_log_theta_1 = tf.log(g1_theta_1)
g1_theta_2 = mu_log_N0*tf.pow(-tf.log(nu[2,0]), 1.0/std_log_N0)
g1_log_theta_2 = tf.log(g1_theta_2)
g1_theta_3 = mu_log_sigma_d*tf.pow(-tf.log(nu[3,0]), 1.0/std_log_sigma_d)
g1_log_theta_3 = tf.log(g1_theta_3)
g1_theta_4 = mu_log_sigma_p*tf.pow(-tf.log(nu[4,0]), 1.0/std_log_sigma_p)
g1_log_theta_4 = tf.log(g1_theta_4)
#g1_theta_5_nontf = poisson_rand(mu_tau)[0]
#g1_theta_5 = tf.Variable(g1_theta_5_nontf)
#g1_log_theta_5 = tf.log(g1_theta_5)
"""

# forward simulation, using plug-ins for u and theta
#x = - tf.log( 1.0 - g2_u ) / g1_theta
x = - tf.log( 1.0 - w ) / g1_theta
# statistics of forward simulation
sx = tf.reduce_mean( x, 1 )
vx = tf.reduce_mean( x*x, 1 ) - sx*sx
stdx = tf.sqrt(vx)

# prior for theta is Gamma( alpha, beta )

#alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
#log_prior_theta = prior_alpha*np.log(prior_beta) + (prior_alpha-1.0)*g1_log_theta  - prior_beta*g1_theta - special.gammaln(prior_alpha)
#log_prior_theta_0 = prior_alpha*np.log(prior_beta) + (prior_alpha-1.0)*g1_log_theta  - prior_beta*g1_theta - special.gammaln(prior_alpha)
#1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)
#prior_theta_0 = 1/(std_log_P * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu_log_P)**2 / (2 * std_log_P**2))
log_prior_theta_0 = -log(std_log_P * np.sqrt(2 * np.pi)) -  ((g1_theta[0] - mu_log_P)**2 / (2 * std_log_P**2))
log_prior_theta_1 = -log(std_log_delta * np.sqrt(2 * np.pi)) -  ((g1_theta[1] - mu_log_delta)**2 / (2 * std_log_delta**2))
log_prior_theta_2 = -log(std_log_N0 * np.sqrt(2 * np.pi)) -  ((g1_theta[2] - mu_log_N0)**2 / (2 * std_log_N0**2))
log_prior_theta_3 = -log(std_log_sigma_d * np.sqrt(2 * np.pi)) -  ((g1_theta[3] - mu_log_sigma_d)**2 / (2 * std_log_sigma_d**2))
log_prior_theta_4 = -log(std_log_sigma_p * np.sqrt(2 * np.pi)) -  ((g1_theta[4] - mu_log_sigma_p)**2 / (2 * std_log_sigma_p**2))
#prior_theta_5 = 

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
#cast no shadow: equation 11 in Blei's VI note after putting P(x,Z) = P(x|Z)P(Z)
#vlb =  log_prior - log_q
objective_function = -vlb 

sess22 = tf.InteractiveSession()
#sess = tf.InteractiveSession()

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

sess22.run(tf.initialize_all_variables())
#sess.run(tf.initialize_all_variables())

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
