# Persunas-Core

本项目作为大数据大作业的核心算法模块，以需求关键词为输入，输出满足需求的用户/岗位画像。项目提供 http 接口供后端调用。

## 推荐开发环境
- 操作系统：Linux
- 语言版本：Python 3.13+
- 编辑器：VSCode
- 包管理器：uv 或 cargo
- 语法检查、格式化：Ruff + Pylance

## 知识图谱的创建

```bash
python src/build_graph.py
```

所得的结果会保存在`./kg_output/`下的csv文件中。