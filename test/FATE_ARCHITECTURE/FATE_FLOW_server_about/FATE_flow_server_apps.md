## fateflow server apps

### 注册apps

```python
def register_page(page_path):
    page_name = page_path.stem.rstrip('_app')
    module_name = '.'.join(page_path.parts[page_path.parts.index('fate_flow'):-1] + (page_name, ))

    spec = spec_from_file_location(module_name, page_path)
    page = module_from_spec(spec)
    page.app = app
    page.manager = Blueprint(page_name, module_name)
    sys.modules[module_name] = page
    spec.loader.exec_module(page)

    page_name = getattr(page, 'page_name', page_name)
    url_prefix = f'/{API_VERSION}/{page_name}'

    app.register_blueprint(page.manager, url_prefix=url_prefix)
    return url_prefix


client_urls_prefix = [
    register_page(path)
    for path in search_pages_path(Path(__file__).parent)
]
scheduling_urls_prefix = [
    register_page(path)
    for path in search_pages_path(Path(__file__).parent.parent / 'scheduling_apps')
]


```


