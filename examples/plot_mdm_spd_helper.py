import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('seaborn')

fontsize=15
matplotlib.rc('font', size=fontsize)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=fontsize)
matplotlib.rc('font',
              family='times',
              serif=['Computer Modern Roman'],
              monospace=['Computer Modern Typewriter'])

EPS=1e-8

class DatasetSPD_2D():

    def __init__(self,n_samples,n_features,n_classes):
        self.n_samples=n_samples
        self.n_features=n_features
        self.n_classes=n_classes
        self.data_helper=DataHelper()

    def generate_sample_dataset(self):
        X,Y=self.easy_gauss()
        X,Y=self.data_helper.shuffle(X,Y)
        return X,Y

    def mat2vec(self,mat):
        n=mat.shape[-1]
        return mat[numpy.tril_indices(n)]

    # returns one random rotation of dimension n
    def random_rotation(self):
        if(self.n_features==2):
            theta=numpy.random.uniform(0,2*numpy.pi)
            c,s=numpy.cos(theta),numpy.sin(theta)
            s=numpy.array([[c,-s],[s,c]])
            return s
        z=(numpy.random.randn(n,n)+1j*numpy.random.randn(n,n))/numpy.sqrt(2.0)
        q,r=numpy.linalg.qr(z)
        d=numpy.diag(r)
        ph=numpy.diag(d/numpy.abs(d))
        # q=numpy.dot(q,numpy.dot(ph,q))
        s=numpy.dot(q,ph)
        return s

    # returns N random rotation matrices close of var_rot around R
    def random_rotations(self,R,var_rot):
        rots=numpy.zeros((self.n_samples,self.n_features,self.n_features))
        for i in range(self.n_samples):
            rots[i]=numpy.dot(R,self.divide_angle_of_cov2(self.random_rotation(),var_rot))
        return rots

    def angle_of_rot2(self,r):
        return numpy.arctan(r[0][1]/r[0][0])

    def divide_angle_of_cov2(self,r,alpha):
        angle=self.angle_of_rot2(r)*alpha
        c,s=numpy.cos(angle),numpy.sin(angle)
        return numpy.array([[c,-s],[s,c]])

    def easy_gauss(self):

        #hyperparams
        M=numpy.random.uniform(-5,5,(self.n_classes,self.n_features)) #get self.n_classes mean mean vectors
        S=numpy.random.uniform(0.1,5.,(self.n_classes,self.n_features)) #get self.n_classes mean diagonal covariances
        if(self.n_features==2):
            R=self.random_rotations(numpy.eye(self.n_features),1) #get self.n_classes mean rotations
        var_mean=numpy.eye(self.n_features)*0.05 #class variance in mean
        var_cov=numpy.eye(self.n_features)*0.05 #in covariance
        if(self.n_features==2):
            var_rot=0.01 #in rotation

        #data
        mu=numpy.zeros((self.n_classes*self.n_samples,self.n_features))
        cov=numpy.zeros((self.n_classes*self.n_samples,self.n_features,self.n_features))
        Y=numpy.zeros((self.n_classes*self.n_samples,self.n_classes))
        for i in range(self.n_classes):
            means=numpy.random.multivariate_normal(M[i],var_mean,self.n_samples)
            covs=numpy.random.multivariate_normal(S[i],var_cov,self.n_samples)
            rots=self.random_rotations(R[i],var_rot)
            mu[i*self.n_samples:(i+1)*self.n_samples]=means
            for j in range(self.n_samples):
                c=numpy.diag(numpy.abs(covs[j]))
                c=numpy.dot(rots[j],numpy.dot(c,rots[j].T))
                cov[i*self.n_samples+j]=c
            Y[i*self.n_samples:(i+1)*self.n_samples,i]=1
        return cov,Y

class PlotHelper():

    def __init__(self):
        self.fig=plt.figure()
        spec=gridspec.GridSpec(ncols=1,nrows=1,figure=self.fig)
        ax=self.fig.add_subplot(spec[0,0])
        self.colors=['r','g','b','xkcd:camel','k']
        self.colors_alt=['xkcd:burgundy','olive','cyan','xkcd:mud brown']

    def plot_ellipse(self,data_point,**kwargs):
        X,Y=self.ellipse(data_point)
        self.fig.axes[0].plot(X,Y,**kwargs)

    def angle_of_rot2(self,r):
        return numpy.arctan(r[0][1]/r[0][0])

    def ellipse(self,P):
        w,vr=numpy.linalg.eig(P)
        w=w.real+EPS
        Np=100

        [e1,e2]=w
        x0,y0=0,0
        angle=self.angle_of_rot2(vr)
        c,s=numpy.cos(angle),numpy.sin(angle)
        the=numpy.linspace(0,2*numpy.pi,Np)
        X=e1*numpy.cos(the)*c-s*e2*numpy.sin(the)+x0
        Y=e1*numpy.cos(the)*s+c*e2*numpy.sin(the)+y0
        return X,Y


class DataHelper():
    '''
    DataHelper provides simple functions to handle data.
    Data is assumed of the following shape:
    X: Data, shape=[n_samples, ...]
    Y: Labels, shape=[n_samples, n_classes] (one-hot encoding)
    '''

    def shuffle(self,X,Y):
        tmp=list(zip(X,Y))
        numpy.random.shuffle(tmp)
        X,Y=zip(*tmp)
        X=numpy.asarray(X)
        Y=numpy.asarray(Y)
        return X,Y

    def get_label_at_index(self,i,labels):
        return numpy.where(labels[i])[0][0]

    def data_with_label(self,data,labels,c):
        return data[numpy.where(numpy.where(labels)[1]==c)]
