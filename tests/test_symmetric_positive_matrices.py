
#Check that SqrtSym(M)=ExpSym(0.5LogSym(M))
def Test():
    Nmax=1000
    err_sqrt=0
    err_invsqrt=0
    for i in range(Nmax):
        M=RndSym()
        if err_sqrt<np.linalg.norm(SqrtSym(M)-ExpSym(0.5*LogSym(M))):
            err_sqrt=np.linalg.norm(SqrtSym(M)-ExpSym(0.5*LogSym(M)))
        if err_invsqrt<np.linalg.norm(InvSqrtSym(M)-ExpSym(-0.5*LogSym(M))):
            err_invsqrt=np.linalg.norm(InvSqrtSym(M)-ExpSym(-0.5*LogSym(M)))
    return('errore sqrt',err_sqrt,'errore InvSqrt:',err_invsqrt)
