import onnx
import onnxruntime

def infer(model_path,input_data):
    #model = onnx.load(model_path)

    # 检查模型是否良好
    #print("Model graph:")
    #print(onnx.get_graph_info(model))

    # 初始化运行时
    ort_session = onnxruntime.InferenceSession(model_path)

    # 准备输入数据
    inputs = ort_session.get_inputs() # 获取模型的输入名称
    input_names=[]
    for i in inputs:
        input_names.append(i.name)
        
    print(input_names)

    # 进行推理
    output_name = ort_session.get_outputs()[0].name # 获取模型的输出名称
    output_names=[]
    for i in ort_session.get_outputs():
        output_names.append(i.name)
    print(output_names)
    for i in input_data:
        print(f"{i}{input_data[i].shape}")
    output_data = ort_session.run(output_names, input_data)
    res={}
    # 输出结果
    for i in range(len(output_names)):
    	res.update({output_names[i]:output_data[i]})
    return res

def cobine2mono(wf):
    assert wf.shape[-1]==2,"The audio has not the streo"
