import os
import time
import csv
import pandas as pd

class RecommendationResult(object):

    def __init__(self,resultPath,epoch):
        self.result_path=resultPath
        self.time_path = time.strftime(
            '%m-%d-%H-%M-%S-', time.localtime(time.time())) + "epoch_"+str(epoch)
        self.root_path=os.path.join(self.result_path,self.time_path)

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=True)
        else:
            raise FileExistsError('{} exists'.format(self.root_path))

        # self.headers=['userId','Top@5','Top@10','Top@20','Top@40','Top@80']
        self.headers = ['userId', 'Top@40']
        if not os.path.exists(os.path.join(self.root_path,'recommendationResult.csv')):
            self.csv_recommendation_result=open(os.path.join(self.root_path, 'recommendationResult.csv'),'w',newline='')
            #新建csv表头列表
            fieldnames=self.headers
            writer=csv.DictWriter(self.csv_recommendation_result,fieldnames=fieldnames)
            #写入表头
            writer.writeheader()
            self.csv_recommendation_result.close()
        #
        self.cnt = 0

    def get_metric_csv_title(self, metrics):
        metric = list(map(lambda x: x.get_title(), metrics))
        return ', '.join(metric)

    def saveRecommendationResult(self,users,metrics):
        # if self.cnt==0:
        #     self.csv_recommendation_result.write('userId, {}\n'.format(
        #          self.get_metric_csv_title(metrics)))
        #     self.cnt+=1
        # count=0
        users=users.tolist()
        result={
            'userId':users
        }
        # for userId in users:
        #     writer.writerow({'userId':userId})

        for i in range(len(metrics)):
            if i %2==0 or metrics[i].topk!=40:
                continue
            tempResult=[]
            metric=metrics[i]
            tempList=metric.recommendation_result.tolist()
            metricTitle=metric.get_title()
            for j in range(len(tempList)):
                #因为tempList中是int所以要转化为string 才能用join
                temp=';'.join('%s' %bundleId for bundleId in tempList[j])                # result.append(temp)
                tempResult.append(temp)
            result[metric.get_title()]=tempResult
        df=pd.DataFrame(result)
        df.to_csv(os.path.join(self.root_path, 'recommendationResult.csv'),header=None,index=None,mode='a')



