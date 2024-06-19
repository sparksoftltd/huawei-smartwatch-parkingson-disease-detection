项目介绍
本项目针对帕金森病人的活动数据采集进行异常检测。
将已有的活动数据通过OCSVM模型筛选出存在的异常数据并剔除，随后将去除异常的数据保存为新的csv文件

模块功能介绍
1、get_newData(data, clf)
该模块的功能是将异常数据去除并保存，
输出拼接好的特征文件。
'''
       Describe
       将异常数据去除并保存
       -----------------
       Parameters
       -----------------
       data:原始特征文件
       clf:OCSVM预训练moxing
       
       Return 活动拼接后的csv文件
       -----------------
'''


执行流程：
1、导入未进行异常处理的文件，并进行数据集划分

       data, x_train, x_test, y_train, y_test = get_Datasets()

3、OCSVM模型训练

    clf = train(x_train) #clf为返回的训练好的模型

4、去除异常并保存数据

     input_data, y_tsne, input_data_new, y_tsne_new = get_newData(data, clf)

     RangeIndex(start=0, stop=2714, step=1)
     已完成去异常后数据和标签拼接保存

5、去除异常前后的可视化

     visual(input_data, y_tsne) #input_data: 数据 y_tsne:标签

