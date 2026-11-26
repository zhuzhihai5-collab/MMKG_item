import sys
#test_Hits@100: 0.7198 test_Hits@10: 0.5430 test_Hits@3: 0.4301 test_Hits@1: 0.3129 test_MR: 483.2426 test_MRR: 0.3909
for line in sys.stdin:
    line=line.strip()
    line=line.split(':')
    ans=[]
    for i,x in enumerate(line):
        if i in [0,5]:continue
        ans.append(x[1:7])
    print('\t'.join(ans))
