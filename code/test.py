import pickle

with open('../ANLI-SNLI.pkl', 'rb') as f:
    data = pickle.load(f)
f.close()

print((data))