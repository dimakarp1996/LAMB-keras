from keras.optimizers import Optimizer

import keras.backend as K
class LAMB(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr, lamb=0.01,beta_1=0.9,
                 beta_2=0.999, eps=1e-6, **kwargs):
        super(LAMB, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, dtype='float32', name='iterations')
            self.lamb =  K.variable(lamb, dtype='float32', name='iterations')
            self.beta_1 = K.variable(beta_1, dtype='float32', name='iterations')
            self.beta_2 = K.variable(beta_2, dtype='float32', name='iterations')
            self.eps= K.variable(eps, dtype='float32', name='iterations')

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        # momentum
        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights  = [self.iterations] + ms + vs 
        for p, g, m,v  in zip(params, grads, ms,vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            mhat_t = m_t /(1 - self.beta_1)
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            vhat_t = v_t/(1-self.beta_2)
            
            self.updates.append(K.update(m_t,m))
            self.updates.append(K.update(v_t,v))
            r1 = K.l2_normalize(p)#, prevweights find example
            r2 = K.l2_normalize(mhat_t / K.pow(vhat_t+self.eps,0.5) + self.lamb*p)
            r = r1/r2
            lr_batch=  r*self.lr
      
            new_p = p - lr_batch*(mhat_t / K.pow(v_t+self.eps,0.5)+ self.lamb*p)
            # find examples. How they update
            if getattr(p, 'constraint', None) is not None:
                        new_p= p.constraint(new_p)
            self.updates.append(K.update(new_p, p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'lambda': float(K.get_value(self.lamb)),
                  'beta1': float(K.get_value(self.beta_1)),
                  'beta2': float(K.get_value(self.beta_2)),
                  'eps':float(K.get_value(self.eps)),
                  }
        base_config = super(LAMB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

