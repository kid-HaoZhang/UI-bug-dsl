DSL for UI bug generate
==============================
此项目定义了一种dsl语法，可以规定任意类型的UI bug且进行对应的生成，主要数据集基于Rico

Archive文件夹为Rico数据集
文件结构为
D:.
├─rico_dataset_v0.1_semantic_annotations
│  └─semantic_annotations
└─unique_uis
    └─combined
main.py可以解析每一条bug.dsl中的规则且生成对应的缺陷数据
TODO标志了待完成内容
