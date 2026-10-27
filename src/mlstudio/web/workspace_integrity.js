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
// Settings contains only editable workspace preferences.
renderSettings=function(){const p=state.preferences||{};return title('Settings','Personalize your workspace.')+`
<div class="preferences-layout">
 <section class="preferences-section" aria-labelledby="settings-appearance">
  <header class="preferences-section-heading"><h2 id="settings-appearance">Appearance</h2><p>Make the workspace feel right for you.</p></header>
  <div class="preference-row"><div class="preference-copy"><label for="pref-theme">Theme</label><p>Choose your workspace colours.</p></div><div class="preference-control"><select id="pref-theme">${[['pastel','Light'],['dark','Dark'],['contrast','High contrast light']].map(([value,label])=>`<option value="${value}" ${p.theme===value?'selected':''}>${label}</option>`).join('')}</select></div></div>
  <div class="preference-row"><div class="preference-copy"><label for="pref-reduce-motion">Reduce motion</label><p>Use quieter transitions and animations.</p></div><label class="preferences-toggle"><input type="checkbox" id="pref-reduce-motion" ${p.reduce_motion?'checked':''}><span aria-hidden="true"></span></label></div>
 </section>
 <section class="preferences-section" aria-labelledby="settings-saving">
  <header class="preferences-section-heading"><h2 id="settings-saving">Saving & storage</h2><p>Keep your projects organised on this device.</p></header>
  <div class="preference-row"><div class="preference-copy"><label for="pref-autosave">Autosave projects</label><p>Save every minute while the workspace is idle. Save a project once to choose its file.</p></div><label class="preferences-toggle"><input type="checkbox" id="pref-autosave" ${p.autosave===true?'checked':''}><span aria-hidden="true"></span></label></div>
  <div class="preference-row preference-storage"><div class="preference-copy"><h3>Storage folder</h3><p class="preferences-path" title="${esc(state.storage)}">${esc(state.storage)}</p>${p.next_storage?`<p class="preferences-restart">After restart: ${esc(p.next_storage)}</p>`:''}<p>Folder changes take effect after restarting. Existing files stay in their current folder.</p></div><div class="preference-control">${button('Change folder','folder','storage_location',false,state.busy)}</div></div>
 </section>
 <details class="preferences-advanced"><summary>Advanced preferences</summary><div class="preference-row"><div class="preference-copy"><label for="pref-budget">Evaluation row budget</label><p>Choose how many records an advanced comparison can use.</p></div><div class="preference-control"><input class="input" id="pref-budget" type="number" min="100" max="20000" step="100" value="${p.research_rows||10000}"></div></div></details>
</div>`;};window.studioPages.settings=renderSettings;
