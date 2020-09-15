import random

#%% 기본 선언
data = [[100,40],
              [200,50],
              [10,5],
              [2,4],
              [120,80],
              [4,10],
              [20,30]]

# 초기 centroid 설정

centroids = random.sample(data, 2)

# 각 군집별 좌표들
clusters = [[], []]

sizeOfCentroids = [0, 0]

def euclidDistance(corr1, corr2):
    return pow(pow((corr1[0]-corr2[0]),2) + pow((corr1[1]-corr2[1]),2),0.5)

def defineClosest(cents, target):
    ret = 1
    if euclidDistance(cents[0], target) < euclidDistance(cents[1], target):
        ret = 2
        sizeOfCentroids[1] += 1
        clusters[1].append(target)
    else:
        sizeOfCentroids[0] += 1
        clusters[0].append(target)
    return ret

def nextCentroid(cluster):
    sums = [0, 0]
    for corr in cluster:
        sums[0] += int(corr[0])
        sums[1] += int(corr[1])
    sums[0] /= len(cluster)
    sums[1] /= len(cluster)
    return sums
    
#%% 실제 작업
    



#%% 결과 출력란


#for i in range(10):
    
print("\n")
print("출력결과")
print("-------------------------------")
print("1 선택데이터 : ", centroids[0])
print("2 선택데이터 : ", centroids[1])
print("-------------------------------")

for corr in data:
    print(corr,"->", defineClosest(centroids, corr))

centroids = [nextCentroid(clusters[0]), nextCentroid(clusters[1])]
    
print("-------------------------------")        
print("1 선택데이터 : 총개수", sizeOfCentroids[0], ", 평균 "
      , round(centroids[0][0], 2), ", ", round(centroids[0][1], 2))
print("2 선택데이터 : 총개수", sizeOfCentroids[1], ", 평균 "
      , round(centroids[1][0], 2), ", ", round(centroids[1][1], 2))
sizeOfCentroids = [0, 0]
cluster = [[], []]

# 1. 초기 좌표 : Initial Centroids
# 2. 과정 진행
# 3. 차기 좌표 : Next Centroids
# 4. 포기 좌표는 빼야됨 -> cluster에 이미 포함