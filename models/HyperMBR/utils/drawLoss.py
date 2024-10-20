# x="epoch:66,item_Teach_bundle_KL_loss:0.004080194514244795,bundle_Teach_item_KL_loss:0.004125823266804218,kd_loss:0.00411213468760252"
from os import path
from re import search
import re
import matplotlib.pyplot as plt


def draw_number(epoch, kd_loss, item_teach_bundle_loss, bundle_teach_item_loss):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    #    plt.bar(step,cost, color="red")
    #    plt.plot(step,cost)
    plt.plot(epoch, kd_loss, color="red", label=r'$L^{ML}$')
    #KL(B||I)是B去拟合I，也就是I teach B
    plt.plot(epoch, item_teach_bundle_loss, color="blue", label=r"KL(B $\parallel$ I)")
    plt.plot(epoch, bundle_teach_item_loss, color="green", label=r"KL(I $\parallel$ B)")
    # plt.plot(step, number_number_all_taskd, color="orange", label="总任务个数")

    plt.legend()  # 显示图例
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.title("loss Change")
    plt.show()  # 画图





def drawLoss(logPath):
    pattern1 = r"epoch:(\d+\.?\d*),item_Teach_bundle_KL_loss:(\d+\.?\d*),bundle_Teach_item_KL_loss:(\d+\.?\d*),kd_loss:(\d+\.?\d*e-?\d*)"
    pattern2 = r"epoch:(\d+\.?\d*),item_Teach_bundle_KL_loss:(\d+\.?\d*),bundle_Teach_item_KL_loss:(\d+\.?\d*),kd_loss:(\d+\.?\d*)"
    epochLists = []
    item_Teach_bundle_KL_loss_Lists = []
    bundle_Teach_item_KL_loss_Lists = []
    kd_loss_Lists = []
    epochList = []
    item_Teach_bundle_KL_loss_List = []
    bundle_Teach_item_KL_loss_List = []
    kd_loss_List = []

    with open(logPath, 'r') as fi:
        for eachLine in fi:
            if eachLine == ">>>>>>>>>>B-I statistics>>>>>>>>>>\n":
                epochList = []
                item_Teach_bundle_KL_loss_List = []
                bundle_Teach_item_KL_loss_List = []
                kd_loss_List = []
            elif eachLine == "Optimization Finished!\n":
                draw_number(epochList, kd_loss_List, item_Teach_bundle_KL_loss_List, bundle_Teach_item_KL_loss_List, )
                epochLists.append(epochList)
                item_Teach_bundle_KL_loss_Lists.append(item_Teach_bundle_KL_loss_List)
                bundle_Teach_item_KL_loss_Lists.append(bundle_Teach_item_KL_loss_List)
                kd_loss_Lists.append(kd_loss_List)
            # eachLine.split(':').split(',')
            result1 = re.findall(pattern1, eachLine)
            result2 = re.findall(pattern2, eachLine)
            if len(result1) != 0 or len(result2) != 0:
                if len(result1) != 0:
                    result = result1[0]
                else:
                    result = result2[0]
                epoch = int(result[0])
                item_Teach_bundle_KL_loss = float(result[1])
                bundle_Teach_item_KL_loss = float(result[2])
                kd_loss = float(result[3])
                if epoch <11:
                    continue
                if epoch < 20 and epoch > 10:
                    kd_loss_List.append(kd_loss * 20 / epoch)
                else:
                    kd_loss_List.append(kd_loss)
                epochList.append(epoch)
                item_Teach_bundle_KL_loss_List.append(item_Teach_bundle_KL_loss)
                bundle_Teach_item_KL_loss_List.append(bundle_Teach_item_KL_loss)
    return epochLists, item_Teach_bundle_KL_loss_Lists, bundle_Teach_item_KL_loss_Lists, kd_loss_Lists




if __name__ == '__main__':
    logFile= '../temperatureChange.log'
    drawLoss(logFile)

    # logFilePath1 = '../withoutML.log'
    # logFilePath2 = '../itemLevelCoefChangewithoutDC.log'
    # epochLists1, item_Teach_bundle_KL_loss_Lists1, bundle_Teach_item_KL_loss_Lists1, kd_loss_Lists1= drawLoss(logFilePath1)
    # epochLists2, item_Teach_bundle_KL_loss_Lists2, bundle_Teach_item_KL_loss_Lists2, kd_loss_Lists2= drawLoss(logFilePath2)
    # for i in range(len(epochLists1)):
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #
    #     #    plt.bar(step,cost, color="red")
    #     #    plt.plot(step,cost)
    #     plt.plot(epochLists1[i], kd_loss_Lists1[i], color="red", label="w/o ML")
    #     plt.plot(epochLists1[i], kd_loss_Lists2[i][:len(epochLists1[i])], color="blue", label="with ML")
    #     # plt.plot(epochLists1, bundle_Teach_item_KL_loss_Lists1, color="green", label="bundle_teach_item_loss")
    #     # plt.plot(epochLists1, bundle_Teach_item_KL_loss_Lists1, color="green", label="bundle_teach_item_loss")
    #
    #
    #     # plt.plot(step, number_number_all_taskd, color="orange", label="总任务个数")
    #
    #     plt.legend()  # 显示图例
    #     plt.xlabel("epoch")
    #     plt.ylabel("loss")
    #     # plt.title("loss Change")
    #     plt.show()  # 画图


