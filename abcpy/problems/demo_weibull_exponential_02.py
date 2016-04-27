import sys;
sys.path.append('/home/tameem/.spyder2')

import numpy as np
import pylab as pp
import scipy as sp
from scipy import special
import tensorflow as tf
from abcpy.problems.exponential import *
from abcpy.plotting import *

#import autograd.numpy as np   # Thinly-wrapped version of Numpy
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad

def taylor_sine(x):  # Taylor approximation to sine function
    ans = currterm = x
    i = 0
    while np.abs(currterm) > 0.001:
        currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans

grad_sine = grad(taylor_sine)
ff = 444;
print "Gradient of sin(pi) is", grad_sine(np.pi)

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
  
def update_tf_list(X, update_index, update_value, total_length):
    tb_updated = tf.slice(X, [update_index, 0], [1, 1])
    tb_before = tf.slice(X, [0, 0], [update_index, 1])
    tb_after = tf.slice(X, [update_index+1, 0], [total_length - 1 - update_index, 1])
    tb_updated = dummy + update_value
    X = tf.concat(0, [tb_before, tb_updated, tb_after])
    return X
    
def update_tf_list_sorted(X, update_index, update_value, total_length):
    
    for i in range(0, update_index):
        #get the correct position
        tobedone=1
    tb_updated = tf.slice(X, [update_index, 0], [1, 1])
    tb_before = tf.slice(X, [0, 0], [update_index, 1])
    tb_after = tf.slice(X, [update_index+1, 0], [total_length - 1 - update_index, 1])
    tb_updated = dummy + update_value
    X = tf.concat(0, [tb_before, tb_updated, tb_after])
    return X
    
def update_tf_matrix_col(X, col_index, col_values, no_rows, no_cols):
    tb_updated = tf.slice(X, [0, col_index], [no_rows, 1])
    tb_before = tf.slice(X, [0, 0], [no_rows, col_index])
    tb_after = tf.slice(X, [0, col_index+1], [no_rows, no_cols - 1 - col_index])
    #tb_updated = tf.assign(tb_updated, col_values)    
    col_values = tf.expand_dims(col_values, 1)
    tb_updated = col_values
    X = tf.concat(1, [tb_before, tb_updated, tb_after])
    return X
    

# Initial Conditions -- some rain drops hit a pond


#X=[]
#X.append[X0]
#for t in range(0, 3):
#    X.append( f(X[-1], parameters) )
#X0 = placeholder()
#theta_var = tf.Variable(2)
S1 = 10
##X=tf.Tensor([S1,T1,2,3])
#X = tf.Variable(tf.zeros([S1, T1], tf.float32))
#N = X
#X[:,t] = f( X[:,t-1])

X = tf.Variable(tf.zeros([S1], tf.float32))
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2])
tf.concat(1, [t1, t2])
r1 = tf.Variable(1)
r2 = tf.Variable(2)
rt = tf.concat(0, [r1, r2])
T1 = 20

t1 = [[11, 2, 3]]
t11 = tf.Variable(t1)
ss = tf.argmin(t11, 1)
sess199 = tf.InteractiveSession()
sess199.run(tf.initialize_all_variables())
total_length = tf.size(t11).eval()
removed_index = ss.eval()[0]
#tf.slice(t11, [removed_index, 0], [1, 1])
tb_before = tf.slice(t11, [0, 0], [1, removed_index])
tb_after = tf.slice(t11, [0, removed_index+1], [1, total_length - 1 - removed_index])
t11 = tf.concat(1, [tb_before, tb_after])

total_length = tf.size(t11).eval()
ss = tf.argmin(t11, 1)
removed_index = ss.eval()[0]
tb_before = tf.slice(t11, [0, 0], [1, removed_index])
if removed_index < (total_length - 1):
    tb_after = tf.slice(t11, [0, removed_index+1], [1, total_length - 1 - removed_index])
    t11 = tf.concat(1, [tb_before, tb_after])
else:
    t11 = tb_before

ss = tf.argmin(t11, 1)
removed_index = ss.eval()[0]
tb_before = tf.slice(t11, [0, 0], [1, removed_index])
tb_after = tf.slice(t11, [0, removed_index+1], [1, total_length - 1 - removed_index])
t11 = tf.concat(1, [tb_before, tb_after])
T1 = 20

##########
S1 = 10
T1 = 1
dummy = tf.Variable(tf.zeros([1.0, 1.0]))
X = tf.Variable(tf.zeros([S1, T1]))

update_index = 4
tb_updated = tf.slice(X, [update_index, 0], [1, 1])
tb_before = tf.slice(X, [0, 0], [update_index, 1])
tb_after = tf.slice(X, [update_index+1, 0], [S1 - 1 - update_index, 1])
tb_updated = dummy + 30.0
X = tf.concat(0, [tb_before, tb_updated, tb_after])

update_index = 7
update_value = 99.0
X = update_tf_list(X, update_index, update_value, S1)

#update_index = 7
#tb_updated = tf.slice(X, [update_index, 0], [1, 1])
#tb_before = tf.slice(X, [0, 0], [update_index, 1])
#tb_after = tf.slice(X, [update_index+1, 0], [S1 - 1 - update_index, 1])
#tb_updated = dummy + 99.0
#X = tf.concat(0, [tb_before, tb_updated, tb_after])

sess199 = tf.InteractiveSession()
sess199.run(tf.initialize_all_variables())
ppp = X.eval()

##########
#dummy = tf.Variable(tf.zeros([1, 1], tf.float32))
#X = tf.Variable(tf.zeros([S1, T1], tf.float32))
#tb_updated = tf.slice(X, [0, 0], [1, 1])
#tb_updated = dummy + 10.0
#rest = tf.Variable(tf.zeros([S1-1, T1], tf.float32))
#X2 = tf.concat(0, [tb_updated, rest])
#sess199 = tf.InteractiveSession()
#sess199.run(tf.initialize_all_variables())
#ppp = X2.eval()


epsilon = 0.25
n = 1
#M = 50
M = 181
nbr_parameters = 5
S = 1

theta_mcmc = "/home/tameem/.spyder2/abcpy/problems/blowfly/theta_MCMC05.txt"
gtruth   = np.loadtxt( theta_mcmc )[:,0]
gtruth2   = np.loadtxt( theta_mcmc )[:,1]
gtruth3   = np.loadtxt( theta_mcmc )[:,2]
gtruth4   = np.loadtxt( theta_mcmc )[:,3]
gtruth5   = np.loadtxt( theta_mcmc )[:,4]

this_mu_log_delta = 2.21
this_sigma_log_delta = 1.07
min_range = this_mu_log_delta - 3.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 3.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')

pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log P$')
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["Posterior V-ABC", "true"])
#pp.show()

#plt.figure(1)
#plt.subplot(211)
#plt.plot(t, s1)
#plt.subplot(212)
#plt.plot(t, 2*s1)
#pp.subplot(2,1,2)
#pp.title("Simulated data")

this_mu_log_delta_pr = 2.5
this_sigma_log_delta_pr = 2
log_model_2 = logofGaussian(fine_theta_range, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range, np.exp(log_model_2), lw=2, ls='--')
pp.hist(gtruth, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper right')
#pp.show()
#pp.savefig("log_P_final.eps", format="eps")

pp.close('all')
#----------------------------------
this_mu_log_delta = -1.49
this_sigma_log_delta = 0.48
min_range = this_mu_log_delta - 7.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 7.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \delta$')
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()
#pp.savefig("foo.eps", format="eps")

this_mu_log_delta_pr = -2
this_sigma_log_delta_pr = 2
log_model_2 = logofGaussian(fine_theta_range, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range, np.exp(log_model_2), lw=2, ls='--')
pp.hist(gtruth2, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper right')
#pp.show()
#pp.savefig("log_delta_final.eps", format="eps")

pp.close('all')
#----------------------------------
this_mu_log_delta = 5.61
this_sigma_log_delta = 0.21
min_range = this_mu_log_delta - 12.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 12.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log N_0$')
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()

this_mu_log_delta_pr = 6
this_sigma_log_delta_pr = 2.9
log_model_2 = logofGaussian(fine_theta_range, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range, np.exp(log_model_2), lw=2, ls='--')
pp.hist(gtruth3, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper right')
#pp.show()
#pp.savefig("log_N0_final.eps", format="eps")

pp.close('all')
#----------------------------------
this_mu_log_delta = -0.34
this_sigma_log_delta = 1.19
min_range = this_mu_log_delta - 6.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 6.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \sigma_d$')
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()

this_mu_log_delta_pr = -1.5
this_sigma_log_delta_pr = 2
log_model_2 = logofGaussian(fine_theta_range, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range, np.exp(log_model_2), lw=2, ls='--')
pp.hist(gtruth4, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left')
#pp.show()
#pp.savefig("log_sigma_d_final.eps", format="eps")

pp.close('all')
#----------------------------------
this_mu_log_delta = -0.33
this_sigma_log_delta = 0.72
min_range = this_mu_log_delta - 7.5 * this_sigma_log_delta
max_range = this_mu_log_delta + 7.5 * this_sigma_log_delta
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title(r'$\log \sigma_p$')
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["V-ABC", "true"])
#pp.show()

this_mu_log_delta_pr = -2.5
this_sigma_log_delta_pr = 2.5
log_model_2 = logofGaussian(fine_theta_range, this_mu_log_delta_pr, this_sigma_log_delta_pr)
pp.plot( fine_theta_range, np.exp(log_model_2), lw=2, ls='--')
pp.hist(gtruth4, normed=True, alpha=0.5)
pp.legend( ["Posterior V-ABC", "Prior", "Rejection sampling"], loc='upper left')
#pp.show()
pp.savefig("log_sigma_p_final.eps", format="eps")
  
def Q_0_nu( S, nbr_parameters ):
  #return np.random.rand( S,1 ).astype('float32')
  return np.random.rand( S, nbr_parameters ).astype('float32')

def Q_0_w( S, M ):
  return np.random.rand( S, M ).astype('float32')
  return np.random.rand( 2, S, M ).astype('float32')
  
#def calc_synth_data( w, g1_th01, g1_th02, g1_th03, g1_th04, g1_th05, tau, index):
def calc_synth_data( w, g1_th, tau, index):
    
    #P       = g1_th01
    #delta   = g1_th02
    #N0      = g1_th03
    #sigma_d = g1_th04
    #sigma_p = g1_th05
    
    T = 180
    #var_d = tf.pow(sigma_d, 2)
    var_d = tf.pow(g1_th[3], 2)
    prec_d = 1.0 / var_d
    
    #var_p = tf.pow(sigma_p, 2)
    var_p = tf.pow(g1_th[4], 2)
    prec_p = 1.0 / var_p
    
    burnin = 50
    lag = int(np.floor(tau))
    if (float(tau)-float(lag)>0.5):
      lag = lag + 1

    N_temp = tf.Variable(tf.zeros([T+1, 1], name = "N_temp"))
    N_temp2 = tf.Variable(tf.zeros([lag+burnin+T, 1], name = "N_temp2"))
    N_temp_sorted = tf.Variable(tf.zeros([T+1, 1], name = "N_temp"))
    #N_temp = np.zeros([lag+burnin+T, 1])
    #N_temp2 = np.zeros(T+1)
    
    sess17 = tf.InteractiveSession()
    sess17.run(tf.initialize_all_variables())
    this_nu = Q_0_nu( S, nbr_parameters )
    this_w = Q_0_w( S, M )
    #N_temp[0] = N0.eval(feed_dict={nu: this_nu, w:this_w })
    N_temp = update_tf_list(N_temp, 0, g1_th[2].eval(feed_dict={nu: this_nu, w:this_w }), T+1)
    #N_temp_sorted = update_tf_list_sorted(N_temp_sorted, 0, g1_th[2].eval(feed_dict={nu: this_nu, w:this_w }), T+1)
    
    for i in range(lag):
        #N_temp[i] = 180.0
        N_temp = update_tf_list(N_temp, i, 180.0, T+1)
        #N_temp_sorted = update_tf_list_sorted(N_temp_sorted, i, 180.0, T+1)
        
    #for i in xrange(burnin+T):
    #for i in xrange(160):
    #for i in xrange(160):
    for i in xrange(60):
        t = i + lag
        
        #eps_t = gamma_rnd( prec_d.eval(feed_dict={nu: this_nu, w:this_w }), prec_d.eval(feed_dict={nu: this_nu, w:this_w }) )
        #e_t   = gamma_rnd( prec_p.eval(feed_dict={nu: this_nu, w:this_w }), prec_p.eval(feed_dict={nu: this_nu, w:this_w }) )
        eps_t = gamma_rnd( prec_d.eval(feed_dict={nu: this_nu, w:this_w }), prec_d.eval(feed_dict={nu: this_nu, w:this_w }) )
        e_t   = gamma_rnd( prec_p.eval(feed_dict={nu: this_nu, w:this_w }), prec_p.eval(feed_dict={nu: this_nu, w:this_w }) )
        #100000000.0        
        #log_eps_t = tf.random_normal([1], -tf.log(var_d + 1)/2 , tf.log(var_d + 1))
        #eps_t = tf.exp(log_eps_t)
        #log_e_t = tf.random_normal([1], -tf.log(var_p + 1)/2 , tf.log(var_p + 1))
        #e_t = tf.exp(log_e_t)
        #if i == 165:
        #    yel = 44
        
        tau_t = t - lag
        #N_temp[t] = P.eval(feed_dict={nu: this_nu, w:this_w })*N_temp[tau_t]*\
        #np.exp(-N_temp[tau_t]/N0.eval(feed_dict={nu: this_nu, w:this_w }))*e_t.eval(feed_dict={nu: this_nu, w:this_w })\
        #+ N_temp[t-1]*np.exp(-delta.eval(feed_dict={nu: this_nu, w:this_w })*eps_t.eval(feed_dict={nu: this_nu, w:this_w }))
        #ahha = N_temp[t-lag,0].eval(feed_dict={nu: this_nu, w:this_w })
        
        
        #ahha = g1_th[0].eval(feed_dict={nu: this_nu, w:this_w })*N_temp[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })*\
        #tf.exp(-N_temp[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })/g1_th[2].eval(feed_dict={nu: this_nu, w:this_w }))*e_t.eval(feed_dict={nu: this_nu, w:this_w })\
        #+ N_temp[t-1,0].eval(feed_dict={nu: this_nu, w:this_w })*tf.exp(-g1_th[1].eval(feed_dict={nu: this_nu, w:this_w })*eps_t.eval(feed_dict={nu: this_nu, w:this_w }))
        #N_temp = update_tf_list(N_temp, t, ahha, T+1)
        ahha1 = g1_th[0].eval(feed_dict={nu: this_nu, w:this_w })*N_temp[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })*\
        tf.exp(-N_temp[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })/g1_th[2].eval(feed_dict={nu: this_nu, w:this_w }))*e_t
        ahha2 = tf.to_float(N_temp[t-1,0].eval(feed_dict={nu: this_nu, w:this_w })*tf.exp(-g1_th[1].eval(feed_dict={nu: this_nu, w:this_w })*eps_t))
        ahha = ahha1 + ahha2
        #ahha = g1_th[0].eval(feed_dict={nu: this_nu, w:this_w })*N_tempouts_list2 = calc_synth_data( w, tf.exp(q_mu_log_theta), poisson_rand(15), 0)[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })*\
        #tf.exp(-N_temp[t - lag,0].eval(feed_dict={nu: this_nu, w:this_w })/g1_th[2].eval(feed_dict={nu: this_nu, w:this_w }))*e_t\
        #+ N_temp[t-1,0].eval(feed_dict={nu: this_nu, w:this_w })*tf.exp(-g1_th[1].eval(feed_dict={nu: this_nu, w:this_w })*eps_t)
        N_temp = update_tf_list(N_temp, t, ahha, T+1)
        #N_temp_sorted = update_tf_list_sorted(N_temp_sorted, t, ahha, T+1)
    
    #N_temp2 = N_temp[-(T+1):]
    #N = tf.Variable(N_temp2)
    #return N_temp2
    
    #outs_list = N_temp.eval()
    #min_range = 1
    #max_range = outs_list.shape[0]
    #theta_range = (min_range, max_range)
    #fine_bin_width = 1
    #fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
    #pp.close('all')
    #pp.figure(1)
    #pp.subplot(1,1,1)
    #pp.title("Simulated data")    
    #pp.plot( fine_theta_range, outs_list, lw=2)
    #pp.show()    
    
    #sort
    for i in range(0, T+1):
        total_length = tf.size(N_temp).eval()
        ss = tf.argmin(N_temp, 0)
        removed_index = ss.eval()[0]
        removed_val = N_temp.eval()[removed_index]
        tb_before = tf.slice(N_temp, [0, 0], [removed_index, 1])
        if removed_index < (total_length - 1):
            tb_after = tf.slice(N_temp, [removed_index+1, 0], [total_length - 1 - removed_index, 1])
            N_temp = tf.concat(0, [tb_before, tb_after])
        else:
            N_temp = tb_before
    N_temp_sorted = update_tf_list(N_temp_sorted, i, removed_val, T+1)
        
    #return N_temp
    return N_temp_sorted


blowfly_filename = "/home/tameem/.spyder2/abcpy/abcpy/problems/blowfly/blowfly.txt"
observations   = np.loadtxt( blowfly_filename )[:,1]
sy = observations.mean()
sorted_observations = np.sort(observations)
sy1 = np.mean( sorted_observations[:observations.size/4] )
sy2 = np.mean( sorted_observations[observations.size/4:observations.size/2] )
#sx3 = tf.reduce_mean( dd[(T+1)/4:3*(T+1)/4,:] )
sy3 = np.mean( sorted_observations[observations.size/2:3*observations.size/4] )
sy4 = np.mean( sorted_observations[3*observations.size/4:] )

##plot y
min_range = 1
max_range = shape(observations)[0]
theta_range = (min_range, max_range)
fine_bin_width = 1
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
#pp.plot( fine_theta_range, observations, lw=2)
#pp.show()   
##End of plot y

#the graph eyah
#blowfly_filename = "/home/tameem/.spyder2/abcpy/problems/bf02.txt"
#sim_obs   = np.loadtxt( blowfly_filename )[:]
#sx = sim_obs.mean()

#min_range = 1
#max_range = shape(sim_obs)[0]
#theta_range = (min_range, max_range)
#fine_bin_width = 1
#fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
#pp.subplot(1,1,1)
#pp.plot( fine_theta_range, observations, lw=2)
#pp.title("Simulated vs. Observed blowfly data")
#pp.xlabel("Time")
#pp.ylabel("Population")
#pp.plot( fine_theta_range, sim_obs, lw=2)
#pp.legend( ["Observed data", "Simulated data"])
#pp.show()
#end of +the graph eyah

q_mu_log_theta = [2.37129045, -1.51626372,  5.65928984, -0.32101333, -0.31801835]
nu = tf.placeholder("float", shape=[S, nbr_parameters], name="nu")
w = tf.placeholder("float", shape=[S, M], name="w")
#outs_list2 = calc_synth_data( w, tf.exp(q_mu_log_theta), poisson_rand(15), 0)
#outs_list_tf = tf.transpose(x)
#outs_list = outs_list_tf.eval(feed_dict={nu: this_nu, w:this_w })

#min_range = 1
#max_range = outs_list2.shape[0]
#theta_range = (min_range, max_range)
#fine_bin_width = 1
#fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
#pp.close('all')
#pp.figure(1)
#pp.subplot(1,2,2)
#pp.title("Simulated data")    
#pp.plot( fine_theta_range, outs_list2, lw=2)
#pp.show()

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

alpha = prior_alpha; beta = prior_beta;
p = ExponentialProblem(alpha, beta)
post_alpha = prior_alpha + M
post_beta  = prior_beta + M*sy

mean_post = post_alpha/post_beta
var_post   = post_alpha/post_beta**2
phi_post_mean = np.log(mean_post)
phi_post_log_std = np.log(np.sqrt(var_post))

#with tf.Session() as sess76:
for ii in range(0, 1):
    count = tf.Variable(0)
    updater = tf.assign(count, count + 1)

    #q_log_alpha_uniform = tf.Variable(0*tf.ones([1]), name="q_log_alpha_uniform")
    #q_log_beta_uniform  = tf.Variable(0*tf.ones([1]), name="q_log_beta_uniform")
    
    #q_weibull_log_scale_lambda = tf.Variable(1.25 + 0*tf.ones([1]), name="q_weibull_log_scale_lambda")
    #q_weibull_log_shape_k = tf.Variable(2*tf.ones([1]), name="q_weibull_log_shape_k")

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
    
    #q_mu_log_P = tf.Variable(2.0, name="q_mu_log_P")
    #q_log_sigma_log_P = tf.Variable(0.6913, name="q_log_sigma_log_P")
    #q_sigma_log_P = tf.exp(0.6913)
    
    #q_mu_log_delta = tf.Variable(-1.0, name="q_mu_log_delta")
    #q_log_sigma_log_delta = tf.Variable(0.6913, name="q_log_sigma_log_delta")
    #q_sigma_log_delta = tf.exp(0.6913)

    #q_mu_log_N0 = tf.Variable(2.0, name="q_mu_log_N0")
    #q_log_sigma_log_N0 = tf.Variable(0.6913, name="q_log_sigma_log_N0")
    #q_sigma_log_N0 = tf.exp(0.6913)
    
    #q_mu_log_sigma_d = tf.Variable(0.0, name="q_mu_log_sigma_d")
    #q_log_sigma_log_sigma_d = tf.Variable(0.6913, name="q_log_sigma_log_sigma_d")
    #q_sigma_log_sigma_d = tf.exp(0.6913)

    #q_mu_log_sigma_p = tf.Variable(0.0, name="q_mu_log_sigma_p")
    #q_log_sigma_log_sigma_p = tf.Variable(0.6913, name="q_log_sigma_log_sigma_p")
    #q_sigma_log_sigma_p = tf.exp(0.6913)
    
    q_mu_log_theta = tf.Variable(tf.zeros([nbr_parameters], tf.float32_ref, name="q_mu_log_theta"))
    q_log_sigma_log_theta = tf.Variable(tf.zeros([nbr_parameters], tf.float32_ref, name="q_log_sigma_log_theta"))
    dummy_mu = np.array([2.0, -1.0, 5.0, 0.0, 0.0])
    #dummy_mu = np.array([0.15, 1.3, 0.1, 1.0, 1.5])
    dummy_log_sigma = np.array([0.6913, 0.6913, 0.6913, 0.6913, 0.6913])
    q_mu_log_theta = q_mu_log_theta + dummy_mu
    q_log_sigma_log_theta = q_log_sigma_log_theta + dummy_log_sigma
    #q_sigma_log_theta = tf.Variable(tf.zeros([nbr_parameters], tf.float32_ref, name="q_sigma_log_theta"))
    #q_mu_log_theta = tf.assign(q_mu_log_theta, np.array([2.0, -1.0, 5.0, 0.0, 0.0]))
    #q_log_sigma_log_theta = tf.assign(q_log_sigma_log_theta, np.array([0.6913, 0.6913, 0.6913, 0.6913, 0.6913]))
    q_sigma_log_theta = tf.exp(q_log_sigma_log_theta)
    dummy_sigma = np.exp(dummy_log_sigma)
    
    #using numpy here only for initialization
    #q_mu_log_theta = tf.to_float(tf.Variable(np.array([2.0, -1.0, 5.0, 0.0, 0.0])))
    #q_log_sigma_log_theta = tf.to_float(tf.Variable(np.array([0.6913, 0.6913, 0.6913, 0.6913, 0.6913])))
    #q_sigma_log_theta = tf.exp(q_log_sigma_log_theta)
    #q_mu_log_theta = tf.Variable(np.array([2.0, -1.0, 5.0, 0.0, 0.0]), name="q_mu_log_theta")
    #q_log_sigma_log_theta = tf.Variable(np.array([0.6913, 0.6913, 0.6913, 0.6913, 0.6913]), name="q_log_sigma_log_theta")
    #q_sigma_log_theta = tf.exp(q_log_sigma_log_theta, name="q_sigma_log_theta")
    
    
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
    
    pr_mu_log_theta = np.array([2.0, -1.0, 5.0, 0.0, 0.0], dtype=float32)
    #pr_mu_log_theta = np.array([0.15, 1.3, 0.1, 1.0, 1.5], dtype=float32)
    pr_sigma_log_theta = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=float32)
    
    #q_weibull_scale_lambda = tf.exp( q_weibull_log_scale_lambda )
    #q_weibull_shape_k = tf.exp( q_weibull_log_shape_k )
    
    # place holders from samples from Q_0
    nu = tf.placeholder("float", shape=[S, nbr_parameters], name="nu")
    w = tf.placeholder("float", shape=[S, M], name="w")
    
    
    #sess888 = tf.InteractiveSession()
    #sess888.run(tf.initialize_all_variables())
    #g1_th2 = tf.zeros([5])
    ###dummy_mu2 = np.array([2.0, -1.0, 5.0, 0.0, 0.0])
    #dummy_mu2 = np.array([np.exp(4.5), np.exp(-1.987), np.exp(5.619), np.exp(0.917), np.exp(1.638)])
    ##dummy_mu2 = np.array([np.exp(2.37187243), np.exp(-1.5007751) , np.exp(5.65928984), np.exp(-0.35911751), np.exp(-0.33275774)])
    #g1_th2 = g1_th2 + dummy_mu2
    #outs_list2 = calc_synth_data( w, g1_th2, poisson_rand(17), 0)
    
    #g1_theta = tf.placeholder("float", shape=[S, nbr_parameters], name="g1_theta")
    #g1_theta = tf.Variable(tf.zeros([S, nbr_parameters]), name = "g1_theta")

    #standbyme = tf.Variable(tf.zeros([S]), name="standbyme")
    
    #g1_log_theta01 = tf.Variable(tf.zeros([S]), name='g1_log_theta01')
    #g1_log_theta02 = tf.Variable(tf.zeros([S]), name='g1_log_theta02')
    #g1_log_theta03 = tf.Variable(tf.zeros([S]), name='g1_log_theta03')
    #g1_log_theta04 = tf.Variable(tf.zeros([S]), name='g1_log_theta04')
    #g1_log_theta05 = tf.Variable(tf.zeros([S]), name='g1_log_theta05')
    #g1_log_theta = tf.Variable(tf.zeros([S, nbr_parameters]), name='g1_log_theta')
    #update_tf_matrix_col(g1_log_theta, 0, q_mu_log_P + q_sigma_log_P*nu[:, 0], S, nbr_parameters)
    #g1_log_theta = q_mu_log_theta + q_sigma_log_theta * nu
    #g1_theta = tf.exp(g1_log_theta)
    g1_theta = tf.exp(-(tf.square(tf.log(nu) - q_mu_log_theta))/(2 * tf.square(q_sigma_log_theta)))/(nu * q_sigma_log_theta * tf.sqrt(2 * np.pi))
    g1_log_theta = tf.log(g1_theta)
    #g1_log_theta01 = q_mu_log_P + q_sigma_log_P*nu[:, 0]
    ####g1_log_theta01 = tf.log(q_mu_log_P*tf.pow(-tf.log(nu[:, 0]), 1.0/q_sigma_log_P))
    #g1_log_theta02 = q_mu_log_delta + q_sigma_log_delta*nu[:, 1]
    #g1_log_theta03 = q_mu_log_N0 + q_sigma_log_N0*nu[:, 2]
    #g1_log_theta04 = q_mu_log_sigma_d + q_sigma_log_sigma_d*nu[:, 3]
    #g1_log_theta05 = q_mu_log_sigma_p + q_sigma_log_sigma_p*nu[:, 4]
    #g1_theta01 = tf.exp(g1_lstdxog_theta01)
    #g1_theta02 = tf.exp(g1_log_theta02)
    #g1_theta03 = tf.exp(g1_log_theta03)
    #g1_theta04 = tf.exp(g1_log_theta04)
    #g1_theta05 = tf.exp(g1_log_theta05)
    with tf.Session() as sess335:
        #g1_theta01_summ = tf.scalar_summary("g1_theta01", g1_theta01)
        #g1_theta02_summ = tf.scalar_summary("g1_theta02", g1_theta02)
        #g1_theta03_summ = tf.scalar_summary("g1_theta03", g1_theta03)
        #g1_theta04_summ = tf.scalar_summary("g1_theta04", g1_theta04)
        #g1_theta05_summ = tf.scalar_summary("g1_theta05", g1_theta05)
        g1_theta_summ = tf.scalar_summary("g1_theta", g1_theta)
        merged = tf.merge_all_summaries()
        #    writer.flush()
  
    T = 180
    mini_S = 1
    #x = tf.Variable(tf.zeros([mini_S, T+1]), name="x")
    x_temp = np.zeros([mini_S, T+1], dtype=np.float32)
    ##for index in range(S):
    #for index in range(mini_S):
    index = 0
    #dd = calc_synth_data(w, g1_theta01[index], g1_theta02[index],g1_theta03[index],g1_theta04[index],g1_theta05[index],poisson_rand(mu_tau),index)
    dd = calc_synth_data(w, g1_theta[index,:],poisson_rand(mu_tau),index)
        #dd = np.reshape(dd, (T+1))
        #x[index, :] = dd
    #x = tf.Variable(x_temp, name="x")
    sx1 = tf.reduce_mean( dd[:(T+1)/4,:] )
    sx2 = tf.reduce_mean( dd[(T+1)/4:(T+1)/2,:] )
    #sx3 = tf.reduce_mean( dd[(T+1)/4:3*(T+1)/4,:] )
    sx3 = tf.reduce_mean( dd[(T+1)/2:3*(T+1)/4,:] )
    sx4 = tf.reduce_mean( dd[3*(T+1)/4:,:] )
    vx1 = tf.reduce_mean(dd*dd) - sx1 * sx1
    vx2 = tf.reduce_mean(dd*dd) - sx2 * sx2
    vx3 = tf.reduce_mean(dd*dd) - sx3 * sx3
    vx4 = tf.reduce_mean(dd*dd) - sx4 * sx4
    stdx1 = tf.sqrt(vx1)
    stdx2 = tf.sqrt(vx2)
    stdx3 = tf.sqrt(vx3)
    stdx4 = tf.sqrt(vx4)
    x = tf.transpose(dd)
    sx = tf.reduce_mean( x, 1 )
    vx = tf.reduce_mean( x*x, 1 ) - sx*sx
    stdx = tf.sqrt(vx)
    
    #q14 = np.mean( sorted[:N/4])
    #q24 = np.mean( sorted[N/4:N/2])
    #q2 = np.mean( sorted[N/4:3*N/4])
    #q34 = np.mean( sorted[N/2:3*N/4]) 
    #q44 = np.mean( sorted[3*N/4:]) 
    #s[0] = np.log(q14/1000.0+1e-12)  #np.log(q1)
    #s[1] = np.log(q24/1000.0+1e-12) #np.log(q24) #np.log(q2)
    #s[2] = np.log(q34/1000.0+1e-12)
    #s[3] = np.log(q44/1000.0+1e-12)    
    #q14 = np.mean( sorted_dif[:N/4]) 
    #q24 = np.mean( sorted_dif[N/4:N/2])
    ##q2 = np.mean( sorted_dif[N/4:3*N/4])
    #q34 = np.mean( sorted_dif[N/2:3*N/4]) 
    #q44 = np.mean( sorted_dif[3*N/4:])    
    #s[4] = q14/1000.0 #np.log(q14+1e-12)  #np.log(q1)
    #s[5] = q24/1000.0 #np.log(q24+1e-12) #np.log(q24) #np.log(q2)
    #s[6] = q34/1000.0 #np.log(q34+1e-12)
    #s[7] = q44/1000.0 #np.log(q44+1e-12)
    
    
    log_prior_theta = np.zeros([nbr_parameters], dtype=np.float32)
    log_prior_theta = -np.log(pr_sigma_log_theta * np.sqrt(2 * np.pi)) -  (np.square(g1_theta - pr_mu_log_theta) / (2.0 * np.square(pr_sigma_log_theta)))
    ###log_prior_theta_1 = tf.Variable(-tf.log(pr_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - pr_mu_log_P,2) / (2.0 * tf.pow(pr_sigma_log_P,2))))
    ###log_prior_theta_1 = -tf.log(pr_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - pr_mu_log_P,2) / (2.0 * tf.pow(pr_sigma_log_P,2)))
    #log_prior_theta_1 = -np.log(pr_sigma_log_P * np.sqrt(2 * np.pi)) -  (np.square(g1_theta01 - pr_mu_log_P) / (2.0 * np.square(pr_sigma_log_P)))
    ###log_prior_theta_2 = -tf.log(pr_sigma_log_delta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta02 - pr_mu_log_delta,2) / (2 * tf.pow(pr_sigma_log_delta,2)))
    #log_prior_theta_2 = -np.log(pr_sigma_log_delta * np.sqrt(2 * np.pi)) -  (np.square(g1_theta02 - pr_mu_log_delta) / (2.0 * np.square(pr_sigma_log_delta)))
    ###log_prior_theta_3 = tf.Variable(-tf.log(pr_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - pr_mu_log_N0,2) / (2 * tf.pow(pr_sigma_log_N0,2))))
    ###log_prior_theta_3 = -tf.log(pr_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - pr_mu_log_N0,2) / (2 * tf.pow(pr_sigma_log_N0,2)))
    #log_prior_theta_3 = -np.log(pr_sigma_log_N0 * np.sqrt(2 * np.pi)) -  (np.square(g1_theta03 - pr_mu_log_N0) / (2.0 * np.square(pr_sigma_log_N0)))
    ###log_prior_theta_4 = tf.Variable(-tf.log(pr_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - pr_mu_log_sigma_d,2) / (2 * tf.pow(pr_sigma_log_sigma_d,2))))
    ###log_prior_theta_4 = -tf.log(pr_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - pr_mu_log_sigma_d,2) / (2 * tf.pow(pr_sigma_log_sigma_d,2)))
    #log_prior_theta_4 = -np.log(pr_sigma_log_sigma_d * np.sqrt(2 * np.pi)) -  (np.square(g1_theta04 - pr_mu_log_sigma_d) / (2.0 * np.square(pr_sigma_log_sigma_d)))
    ###log_prior_theta_5 = tf.Variable(-tf.log(pr_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - pr_mu_log_sigma_p,2) / (2 * tf.pow(pr_sigma_log_sigma_p,2))))
    ###log_prior_theta_5 = -tf.log(pr_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - pr_mu_log_sigma_p,2) / (2 * tf.pow(pr_sigma_log_sigma_p,2)))
    #log_prior_theta_5 = -np.log(pr_sigma_log_sigma_p * np.sqrt(2 * np.pi)) -  (np.square(g1_theta05 - pr_mu_log_sigma_p) / (2.0 * np.square(pr_sigma_log_sigma_p)))
        
    # prior for u is uniform(0,1)
    log_prior_u = 0.0
        
    #log_q_theta = tf.Variable(tf.zeros([S, nbr_parameters]), name = "log_q_theta")
    log_q_theta = -tf.log(q_sigma_log_theta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta - q_mu_log_theta,2) / (2 * tf.pow(q_sigma_log_theta,2)))
    #log_const = q_sigma_log_theta*(g1_log_theta - tf.log(q_mu_log_theta))
    #log_q_theta = q_log_sigma_log_theta - q_sigma_log_theta*tf.log(q_mu_log_theta) + (q_sigma_log_theta-1.0)*g1_log_theta - tf.exp(log_const)
    ###log_q_theta_1 = tf.Variable(-tf.log(q_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - q_mu_log_P,2) / (2 * tf.pow(q_sigma_log_P,2))))
    #log_q_theta_1 = -tf.log(q_sigma_log_P * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta01 - q_mu_log_P,2) / (2 * tf.pow(q_sigma_log_P,2)))
    ###log_q_theta_2 = tf.Variable(-tf.log(q_sigma_log_delta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta02 - q_mu_log_delta,2) / (2 * tf.pow(q_sigma_log_delta,2))))
    #log_q_theta_2 = -tf.log(q_sigma_log_delta * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta02 - q_mu_log_delta,2) / (2 * tf.pow(q_sigma_log_delta,2)))
    ###log_q_theta_3 = tf.Variable(-tf.log(q_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - q_mu_log_N0,2) / (2 * tf.pow(q_sigma_log_N0,2))))
    #log_q_theta_3 = -tf.log(q_sigma_log_N0 * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta03 - q_mu_log_N0,2) / (2 * tf.pow(q_sigma_log_N0,2)))
    ###log_q_theta_4 = tf.Variable(-tf.log(q_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - q_mu_log_sigma_d,2) / (2 * tf.pow(q_sigma_log_sigma_d,2))))
    #log_q_theta_4 = -tf.log(q_sigma_log_sigma_d * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta04 - q_mu_log_sigma_d,2) / (2 * tf.pow(q_sigma_log_sigma_d,2)))
    ###log_q_theta_5 = tf.Variable(-tf.log(q_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - q_mu_log_sigma_p,2) / (2 * tf.pow(q_sigma_log_sigma_p,2))))
    #log_q_theta_5 = -tf.log(q_sigma_log_sigma_p * tf.pow(2 * np.pi, 0.5)) -  (tf.pow(g1_theta05 - q_mu_log_sigma_p,2) / (2 * tf.pow(q_sigma_log_sigma_p,2)))
    
    eps_x = stdx/np.sqrt(M)
    eps_x1 = stdx1/np.sqrt(M)
    eps_x2 = stdx2/np.sqrt(M)
    eps_x3 = stdx3/np.sqrt(M)
    eps_x4 = stdx4/np.sqrt(M)
    #log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x)-0.5*tf.pow(sy - sx,2.0)/tf.pow(eps_x,2) )
    log_likelihood = tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x1)-0.5*tf.pow(sy1 - sx1,2.0)/tf.pow(eps_x1,2) ) +\
    tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x2)-0.5*tf.pow(sy2 - sx2,2.0)/tf.pow(eps_x2,2) )+\
    tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x3)-0.5*tf.pow(sy3 - sx3,2.0)/tf.pow(eps_x3,2) )+\
    tf.reduce_mean( -0.5*np.sqrt(2*np.pi)-tf.log(eps_x4)-0.5*tf.pow(sy4 - sx4,2.0)/tf.pow(eps_x4,2) )
    log_prior = tf.reduce_mean( log_prior_theta )
    log_q     = tf.reduce_mean( log_q_theta )
    #log_prior_1 = tf.reduce_mean( log_prior_theta_1 )
    #log_prior_2 = tf.reduce_mean( log_prior_theta_2 )
    #log_prior_3 = tf.reduce_mean( log_prior_theta_3 )
    #log_prior_4 = tf.reduce_mean( log_prior_theta_4 )
    #log_prior_5 = tf.reduce_mean( log_prior_theta_5 )
    #log_q_1     = tf.reduce_mean( log_q_theta_1 )
    #log_q_2     = tf.reduce_mean( log_q_theta_2 )
    #log_q_3     = tf.reduce_mean( log_q_theta_3 )
    #log_q_4     = tf.reduce_mean( log_q_theta_4 )
    #log_q_5     = tf.reduce_mean( log_q_theta_5 )

    # variational lower bound
    #vlb = log_likelihood + log_prior_1 + log_prior_2 + log_prior_4 + log_prior_5 - log_q_1 - log_q_2 - log_q_4 - log_q_5
    vlb = log_likelihood + log_prior - log_q
    #vlb = log_likelihood + log_prior_1 - log_q_1
    #vlb = log_likelihood - log_q_1
    objective_function = -vlb

    sess88 = tf.InteractiveSession()
    
    print "========================================"
    print "1- check the prior is correct, why so off from true posterior? "
    print "2- kl divergence with true posterior"
    print "3- thetas from Q_phi(theta) have high prob under prior"
    print "4- mistakes converting gamma alpha/beta to lognormal means/vars"
    print "5- truncated normal instead of lognormal"
    print "6- variational for long tailed posterior, gamma?"
    print "7- non-rejection based generator for gamma? -- see wikipedia"
    print "========================================"

    #train_step = tf.train.AdamOptimizer(5*1e-2).minimize(objective_function)
    train_step = tf.train.AdamOptimizer(1*1e-3).minimize(objective_function)
    #5*1e-4
    
    sess88.run(tf.initialize_all_variables())

    #this_nu = Q_0_nu( S, nbr_parameters )
    #this_w = Q_0_w( S, M )    
    #sess76.run(tf.initialize_all_variables(), feed_dict={nu: this_nu, w:this_w })
    
    #for i in range(1500):
    for i in range(500):
        this_nu = Q_0_nu( S, nbr_parameters )
        this_w = Q_0_w( S, M )
        #sess76.run(train_step, feed_dict={nu: this_nu, w:this_w })
        train_step.run(feed_dict={nu: this_nu, w:this_w } )        
        
        if i%10 == 0:
            ###this_scale_lambda = q_weibull_scale_lambda.eval()[0]
            ###this_shape_k = q_weibull_shape_k.eval()[0]
            #this_mu_log_P = q_mu_log_P.eval()
            #this_sigma_log_P = q_sigma_log_P.eval()
            #this_mu_log_delta = q_mu_log_delta.eval()
            #this_sigma_log_delta = q_sigma_log_delta.eval()
            #this_mu_log_N0 = q_mu_log_N0.eval()
            #this_sigma_log_N0 = q_sigma_log_N0.eval()
            #this_mu_log_sigma_d = q_mu_log_sigma_d.eval()
            #this_sigma_log_sigma_d = q_sigma_log_sigma_d.eval()
            #this_mu_log_sigma_p = q_mu_log_sigma_p.eval()
            #this_sigma_log_sigma_p = q_sigma_log_sigma_p.eval()
            this_mu_log_theta = q_mu_log_theta.eval()
            this_sigma_log_theta = q_sigma_log_theta.eval()
            
            #print log_q.eval(feed_dict={nu: this_nu, w:this_w })
            this_sx = sx.eval(feed_dict={nu: this_nu, w:this_w })
            this_log_prior = log_prior.eval(feed_dict={nu: this_nu, w:this_w })
            this_log_q = log_q.eval(feed_dict={nu: this_nu, w:this_w })
            this_ll = log_likelihood.eval(feed_dict={nu: this_nu, w:this_w })
            obj_func = objective_function.eval(feed_dict={nu: this_nu, w:this_w })
            #print "tensorflow -- step %d, objective_function %g, mean_x %g +- %g  mean_y %g  log_prior %g  log_q %g  lambda %g k %g"%(i, \
            #obj_func , this_sx.mean(), this_sx.std(), sy,this_log_prior_1, this_log_q_1,  this_scale_lambda, this_shape_k )
            print "tensorflow -- step %d, objective_function %g, mean_x %g +- %g  mean_y %g  log_prior %g  log_q %g LL %g mu_N0 %g sigma_N0 %g"%(i, \
            obj_func , this_sx.mean(), this_sx.std(), sy,this_log_prior, this_log_q, this_ll, this_mu_log_theta[2], this_sigma_log_theta[2])
            dfd = 5
            #outs_list_tf = tf.transpose(x)
            #outs_list = outs_list_tf.eval(feed_dict={nu: this_nu, w:this_w })
            #min_range = 1
            #max_range = outs_list.shape[0]
            #theta_range = (min_range, max_range)
            #fine_bin_width = 1
            #fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
            #pp.close('all')
            #pp.figure(1)
            #pp.subplot(1,1,1)
            #pp.title("blowfly data")    
            #pp.plot( fine_theta_range, outs_list, lw=2)
            #pp.show()
            ##g1_th2 = tf.zeros([nbr_parameters])
            ##dummy_mu2 = np.array([2.0, -1.0, 5.0, 0.0, 0.0])
            ##g1_th2 = g1_th2 + dummy_mu
            ##outs_list2 = calc_synth_data( w, g1_th2, poisson_rand(15), 0)
            
    enough = 7
    outs_list2 = calc_synth_data( w, tf.exp(q_mu_log_theta), poisson_rand(15), 0)
    #outs_list_tf = tf.transpose(x)
    #outs_list = outs_list_tf.eval(feed_dict={nu: this_nu, w:this_w })
    min_range = 1
    max_range = outs_list2.shape[0]
    theta_range = (min_range, max_range)
    fine_bin_width = 1
    fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
    pp.close('all')
    pp.figure(1)
    pp.subplot(1,1,1)
    pp.title("blowfly data")    
    pp.plot( fine_theta_range, outs_list2, lw=2)
    pp.show()
dummy = 42


for kk in range(0, 5):
    this_mu_log_delta = q_mu_log_theta.eval()[kk]
    this_sigma_log_delta = q_sigma_log_theta.eval()[kk]
    min_range = this_mu_log_delta - 2.5 * this_sigma_log_delta
    max_range = this_mu_log_delta + 2.5 * this_sigma_log_delta
    theta_range = (min_range, max_range)
    fine_bin_width = 0.005
    fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
    pp.close('all')
    pp.figure(1)
    pp.subplot(1,1,1)
    pp.title("log_delta")
    log_model_1 = logofGaussian(fine_theta_range, this_mu_log_delta, this_sigma_log_delta)
    pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
    pp.legend( ["vb", "true"])
    pp.show()
enough = 1




min_range = this_mu_log_P - 2 * this_sigma_log_P
max_range = this_mu_log_P + 2 * this_sigma_log_P
theta_range = (min_range, max_range)
fine_bin_width = 0.005
fine_theta_range = np.arange(min_range, max_range+fine_bin_width, fine_bin_width)
pp.close('all')
pp.figure(1)
pp.subplot(1,1,1)
pp.title("log_P")
log_model_1 = logofGaussian(fine_theta_range, this_mu_log_P, this_sigma_log_P)

pp.plot( fine_theta_range, np.exp(log_model_1), lw=2)
pp.legend( ["vb", "true"])
pp.show()
enough = 1