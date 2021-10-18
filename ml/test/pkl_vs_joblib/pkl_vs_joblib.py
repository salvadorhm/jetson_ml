import joblib
import time
import numpy
import pickle

bigarray = numpy.zeros([190,91,190])
bigarray = bigarray.flatten()


### Saving
start = time.time()
joblib.dump(bigarray,"bigarray1.joblib")
end = time.time() - start
print(end)


start = time.time()
pickle.dump(bigarray,open("bigarray2.pkl","wb"))
end = time.time()-start
print(end)


### Loading
start = time.time()
joblib.load("bigarray1.joblib")
end = time.time() - start
print(end)



start = time.time()
pickle.load(open("bigarray2.pkl","rb"))
end = time.time()-start
print(end)

