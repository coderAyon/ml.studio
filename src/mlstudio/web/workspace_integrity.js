// Dataset identity, selection and capability information shared across the workspace.
let integrityInstance='';
const integrityRender=render;
render=function(){
 const instance=state?.dataset?.instance_key||'';
 if(instance!==integrityInstance){integrityInstance=instance;explorer.page=0;explorer.sort='';explorer.descending=false;explorer.column='';explorer.selected.clear();explorer.hidden.clear();view.search='';outlierUI.selected=null;}
 integrityRender();
};
const integrityData=renderData;
renderData=function(){const t=document.createElement('template');t.innerHTML=integrityData();const toolbar=t.content.querySelector('#dataset-search')?.closest('.toolbar');if(toolbar&&explorer.selected.size)toolbar.insertAdjacentHTML('beforeend',`<button class="button small" data-integrity-selection ${state.busy?'disabled':''}>Preview keeping ${explorer.selected.size} selected rows</button><button class="text-link" data-integrity-clear>Clear selection</button>`);return t.innerHTML;};window.studioPages.data=renderData;
document.addEventListener('click',e=>{if(e.target.closest('[data-integrity-clear]')){explorer.selected.clear();render();}if(e.target.closest('[data-integrity-selection]')){const positions=Array.from(explorer.selected).sort((a,b)=>a-b).map(p=>p-1);navigate('preprocessing');cmd('preprocess_preview',{operation:'Select row positions',column:'',value:JSON.stringify(positions)});}});
const integritySettings=renderSettings;
renderSettings=function(){const t=document.createElement('template');t.innerHTML=integritySettings();t.content.querySelectorAll('.settings-row').forEach(row=>{if(row.textContent.includes('What’s still growing'))row.innerHTML='<div><h3>Available tools and current limits</h3><p>Save/load fitted .mlmodel files, portable project snapshots, 38 workflow steps, five chart types, preprocessing and supervised/clustering experiments are available.</p><p>Train & compare: 8–20,000 records, at most 200 retained inputs. Workflow and preparation previews: up to 20,000 records / 300 source columns. Clustering: 3–10,000 records / 100 numeric inputs. KNN filling: 2,000 records / 50 numeric inputs. Prediction CSV: 20,000 records / 300 columns. Table pages show 100 rows; search and outlier counts use the full active dataset.</p><p>Deep Learning/GPU, arbitrary Python/R plugins and complete Orange/Weka extension coverage are not implemented. The project-root EXE uses this folder and its Python environment; clean-machine portable acceptance remains separate.</p></div>';});return t.innerHTML;};window.studioPages.settings=renderSettings;
