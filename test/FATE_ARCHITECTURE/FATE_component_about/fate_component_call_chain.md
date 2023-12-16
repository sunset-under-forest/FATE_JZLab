## fate_flow组件（算法）调用流程


### 启动fate_flow_server过程，在FATE/fateflow/python/fate_flow/components/components.py下的_search_components添加print(traceback.format_stack())
```
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/fate_flow_server.py", line 89, in <module>
    default_algorithm_provider = ProviderManager.register_default_providers()
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/manager/provider_manager.py", line 35, in register_default_providers
    code, result = cls.register_fate_flow_provider()
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/manager/provider_manager.py", line 46, in register_fate_flow_provider
    return WorkerManager.start_general_worker(worker_name=WorkerName.PROVIDER_REGISTRAR, provider=provider, run_in_subprocess=False)
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/manager/worker_manager.py", line 143, in start_general_worker
    code, message, result = module().run(**kwargs)
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/worker/base_worker.py", line 142, in run
    result = self._run()
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/worker/provider_registrar.py", line 29, in _run
    support_components = ComponentRegistry.register_provider(provider)
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/db/component_registry.py", line 47, in register_provider
    support_components = provider_interface.get_names()
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/components/components.py", line 68, in get_names
    obj_pairs, module_name = _search_components(p.resolve(), cls._module_base())
  File "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/components/components.py", line 47, in _search_components
    print(traceback.format_stack())
```

调用链：
![component_call.svg](./images%2Fcomponent_call.svg)


### 动态调试代码

```python 
with open("test_log.txt", "a") as f:
     print("debug!"  , file=f)
     print(xxx,file=f)
     print("debug finish!"  , file=f)

```

```python
    os.system('echo "debug!" >> test_log.txt')
    os.system('sqlite3 ../fate_sqlite.db "SELECT * FROM t_component_info" >> test_log.txt')
```