# FATE框架理解

这里记录的是我们团队对FATE框架的研究

研究时间
- 2023年7月——现在

研究内容包括但不限于FATE框架的（以下链接可直接导向相关文档）：
- [项目结构（目录结构，各个目录功能作用）](./FATE_project_basic/FATE_basic_intro.md)
- [底层通信实现方式（federation包）](./FATE_communication_analyze/FATE_communication_analyze.md)
- [底层安全协议实现方式（secureprotol包）](./FATE_secureprotol)
- [数据底层操作（数据库交互）](./FATE_database_analyze/fate_database_analyze.md)
- [表操作（存储表，计算表）](./FATE_table_about)
- [组件注册过程](./FATE_component_about/fate_component_register_process.md)
- [组件调用过程](./FATE_component_about/fate_component_call_chain.md)
- [组件运行过程](./FATE_component_about/FATE_component_run_analyze_in_detail.md)
- [命令行工具原理（flow，pipeline）](./FATE_client/FATE_flow_client.md)
- [fateflow server（启动入口点，相关接口作用）](./FATE_FLOW_server_about/FATE_flow_server_apps.md)
- [任务调度（job）](./FATE_job_about/FATE_job_create_process.md)
- [联邦学习过程（线性回归组件详细分析）](./FATE_component_about/FATE_hetero_sshe_linear_component_analyze.md)

研究方法包括但不限于：
- 静态源码分析（大部分源码逐行分析）
- 动态调试（内核代码不方便使用调试器，只能手动打桩）
- 手动连接数据库查看数据（sqlite3 连接）
- 自定义数据集，运行参数进行测试
- 设计实现一系列测试组件