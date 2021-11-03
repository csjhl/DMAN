import torch

# M = 2
# D = 2
# T = 3
# je = torch.tensor(list(range(M*D))).reshape(M,D)
# w = torch.tensor(list(range(M*(T)*D*D))).reshape(M,T,D,D)
# ie = torch.tensor(list(range((T)*D))).reshape(T, D)
#
# res2 = torch.einsum('id, itdk, tk -> it',je, w, ie)
# print(res2)
# print(res2.shape)
#
# tmp1 = torch.zeros(M,T,D)
# for i in range(M):
#     for j in range(T):
#         tmp1[i,j] = je[i] @ w[i,j]
#
# res = torch.zeros(M,T)
# for i in range(M):
#     for j in range(T):
#         # print(ie[j],ie[j].shape,ie[j].t())
#         res[i,j] = (tmp1[i,j] @ ie[j].t().float())
#
# print(res)
# print(res.shape)

a = torch.tensor([[0,0,0],[-567651,999999999999999,99999999]]).float()
b = torch.exp(a)
c = a/b
d = torch.norm(c,dim=-1,keepdim=True)
e = torch.sigmoid(a)
f = a.norm(dim=-1)
print(a)
print(b)
print(c)
print(d)
print(b/b)
print(e)
print(f)
print((f*f)/(1+f*f))