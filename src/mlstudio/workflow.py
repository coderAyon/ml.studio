"""Restricted, versioned directed graphs with explicit port validation."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from .data import Transformation
from .training import train,compare,REGRESSION

PORTS = {"Dataset": (None, "data"), "Remove duplicates": ("data", "data"), "Fill missing": ("data", "data"), "Train model": ("data", "result")}
PORTS.update({'Prepare training':('data','data'),'Compare models':('data','result')})
PORTS.update({kind:('data','data') for kind in ['Data table','Select columns','Impute missing','Normalize']})
PORTS.update({kind:('data','view') for kind in ['Prepared output','Histogram','Scatter plot','Box plot','Bar chart','Correlation']})
PORTS['Results']=('result','result')
PORTS.update({kind:('data','data') for kind in ['Filter rows','Sample rows','Drop missing rows','Rename column','Convert numeric','Trim whitespace','PCA','Select features','Discretize']})
PORTS.update({kind:('data','view') for kind in ['Statistics','Outlier review']})
PORTS.update({kind:('data','data') for kind in ['K-Means','DBSCAN','Hierarchical']})
PORTS.update({kind:('data','data') for kind in ['Merge data','Concatenate','Group by','Pivot table']})
MULTI_INPUTS = {'Merge data', 'Concatenate'}
PORTS.update({kind:('data','result') for kind in ['Decision Tree','Random Forest','SVM','KNN','Naive Bayes','Linear Regression','Ridge','Random Forest Regressor']})


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def order(self,allow_disconnected=False):
        if not self.nodes:
            raise ValueError("Add a Dataset node before validating or running a workflow.")
        incoming = {key: 0 for key in self.nodes}
        for key, node in self.nodes.items():
            if node.get("type") not in PORTS:
                raise ValueError(f"Unavailable node type: {node.get('type')}")
        for a, b in self.edges:
            if a not in self.nodes or b not in self.nodes or a == b:
                raise ValueError("Invalid connection endpoint.")
            if PORTS[self.nodes[a]["type"]][1] != PORTS[self.nodes[b]["type"]][0]:
                raise ValueError("Incompatible port types.")
            incoming[b] += 1
            capacity=2 if self.nodes[b]['type'] in MULTI_INPUTS else 1
            if incoming[b] > capacity:
                raise ValueError(f'This input accepts at most {capacity} connection(s).')
        for key, degree in incoming.items():
            if PORTS[self.nodes[key]["type"]][0] and degree == 0 and not allow_disconnected:
                raise ValueError("A required data input is disconnected.")
            if self.nodes[key]['type'] in MULTI_INPUTS and degree != 2 and not allow_disconnected:
                raise ValueError('Connect two tables to this step.')
        ready = [key for key in incoming if incoming[key] == 0]
        order = []
        while ready:
            key = ready.pop(0)
            order.append(key)
            for a, b in self.edges:
                if a == key:
                    incoming[b] -= 1
                    if incoming[b] == 0:
                        ready.append(b)
        if len(order) != len(self.nodes):
            raise ValueError("Workflow contains a cycle.")
        return order

    def execute(self, frame, target, algorithm, preparation=None):
        supported={'Dataset','Data table','Remove duplicates','Fill missing','Prepare training',
                   'Train model','Compare models',*MULTI_INPUTS,'Group by','Pivot table'}
        order=self.order()
        for key in order:
            if self.nodes[key]['type'] not in supported:
                raise ValueError(f"Use the reactive workflow canvas to execute {self.nodes[key]['type']}; this legacy executor does not implement it.")
        results = {};summarized={}
        for key in order:
            kind = self.nodes[key]["type"]
            parents = [a for a, b in self.edges if b == key]
            summarized[key]=kind in {'Group by','Pivot table'} or any(summarized[a] for a in parents)
            if kind in {'Train model','Compare models'} and summarized[key]:
                raise ValueError('Train and evaluate from a branch before grouping or pivoting.')
            data = results[parents[0]] if parents else frame.copy(deep=True)
            if kind == "Dataset":
                results[key] = data
            elif kind == 'Data table':
                results[key] = data.copy(deep=True)
            elif kind in MULTI_INPUTS:
                from .table_operations import combine_tables
                results[key]=combine_tables(results[parents[0]],results[parents[1]],kind,self.nodes[key].get('params',{}))
            elif kind in {'Group by','Pivot table'}:
                from .table_operations import summarize_table
                results[key]=summarize_table(data,kind,self.nodes[key].get('params',{}))
            elif kind == "Train model":
                results[key] = train(data, target, algorithm,preparation=preparation)
            elif kind=='Prepare training':
                from .training_prep import validate_preparation
                validate_preparation(preparation,data.columns,target)
                results[key]=data  # Learned preparation is intentionally deferred until each fit.
            elif kind=='Compare models':
                algorithms=['Ridge','Decision Tree Regressor','Random Forest Regressor'] if algorithm in REGRESSION else ['Logistic Regression','Decision Tree','Random Forest']
                results[key]=compare(data,target,algorithms,preparation=preparation)
            elif kind == "Fill missing":
                raise ValueError("Exploratory imputation cannot precede evaluation. Training already fits imputation within the training split.")
            elif kind == 'Remove duplicates':
                results[key] = Transformation("Remove duplicates", []).preview(data)
        return results

    def save(self, path):
        Path(path).write_text(json.dumps({"version": 1, "nodes": self.nodes, "edges": self.edges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("nodes"), dict) or not isinstance(payload.get("edges"), list):
            raise ValueError("Unsupported workflow schema.")
        graph = cls(payload["nodes"], payload["edges"])
        graph.order()
        return graph
