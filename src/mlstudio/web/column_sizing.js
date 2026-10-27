// Column widths belong to the current dataset view, never to the CSV or model inputs.
const columnSize={instance:'',widths:new Map(),selected:'',drag:null,min:72,max:1000,default:160};
function columnWidth(name){return columnSize.widths.get(name)??columnSize.default;}
function columnMeasure(table,name){
 const headers=Array.from(table.querySelectorAll('thead th')),index=headers.findIndex(th=>th.dataset.widthColumn===name);if(index<0)return columnSize.default;
 const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d'),style=getComputedStyle(table);ctx.font=`${style.fontSize} ${style.fontFamily}`;
 let size=ctx.measureText(name).width+44;
 Array.from(table.tBodies[0]?.rows||[]).slice(0,100).forEach(row=>{size=Math.max(size,ctx.measureText(row.cells[index]?.textContent||'').width+36);});
 return Math.max(columnSize.min,Math.min(500,Math.ceil(size)));
}
function columnApply(table){
 const cols=Array.from(table.querySelectorAll('colgroup col'));let total=0;
 cols.forEach((col,i)=>{const width=i===0?70:i===1?65:columnWidth(col.dataset.widthColumn);col.style.width=width+'px';total+=width;});
 table.style.width=total+'px';table.style.minWidth=total+'px';
 table.querySelectorAll('[data-column-grip]').forEach(handle=>handle.setAttribute('aria-valuenow',columnWidth(handle.dataset.columnGrip)));
 const control=document.querySelector('#column-size-value');if(control)control.value=columnWidth(columnSize.selected);
 const nameControl=document.querySelector('#column-size-name');if(nameControl)nameControl.value=columnSize.selected;
}
const sizedData=renderData;
renderData=function(){
 const t=document.createElement('template');t.innerHTML=sizedData();const d=state.dataset;if(!d)return t.innerHTML;
 if(columnSize.instance!==d.instance_key){columnSize.instance=d.instance_key;columnSize.widths.clear();columnSize.selected='';columnSize.drag=null;document.body.classList.remove('column-resizing');}
 const panel=t.content.querySelector('.easy-dataset-table'),table=panel?.querySelector('.table-scroll table');if(!table)return t.innerHTML;
 const names=d.fields.map(f=>f.name),visible=names.filter(n=>!explorer.hidden.has(n));if(!names.includes(columnSize.selected))columnSize.selected=visible[0]||names[0]||'';
 table.classList.add('scalable-table');const group=document.createElement('colgroup');
 Array.from(table.querySelectorAll('thead th')).forEach((th,i)=>{
  const name=th.querySelector('[data-sort-column]')?.dataset.sortColumn,col=document.createElement('col');
  if(name!==undefined){col.dataset.widthColumn=name;th.dataset.widthColumn=name;th.insertAdjacentHTML('beforeend',`<span class="column-resize-handle" data-column-grip="${esc(name)}" role="separator" tabindex="0" aria-orientation="vertical" aria-label="Resize ${esc(name)}" aria-valuemin="${columnSize.min}" aria-valuemax="${columnSize.max}" aria-valuenow="${columnWidth(name)}" title="Drag to resize · double-click to fit · arrow keys for fine adjustment"></span>`);}
  const width=i===0?70:i===1?65:columnWidth(name);col.style.width=width+'px';group.append(col);
 });
 table.prepend(group);const total=135+visible.reduce((sum,n)=>sum+columnWidth(n),0);table.style.width=total+'px';table.style.minWidth=total+'px';
 const settings=document.createElement('details');settings.className='column-size-settings';settings.id='column-size-settings';settings.innerHTML=`<summary>Column widths · drag a header edge to resize</summary><div class="column-size-controls"><label>Column<select id="column-size-name">${names.map(n=>`<option value="${esc(n)}" ${columnSize.selected===n?'selected':''}>${esc(n)}</option>`).join('')}</select></label><label>Width (px)<input class="input" id="column-size-value" type="number" min="${columnSize.min}" max="${columnSize.max}" value="${columnWidth(columnSize.selected)}"></label><button class="button small" data-column-size="apply">Apply width</button><button class="button small" data-column-size="compact">Compact columns</button><button class="button small" data-column-size="fit">Auto-fit visible</button><button class="text-link" data-column-size="reset">Reset widths</button><p>72–1,000 px per column. Auto-fit measures the current page, up to 500 px. Widths stay while you sort, filter, change pages or hide columns. Scroll sideways for additional columns.</p></div>`;
 panel.querySelector('.easy-table-note')?.before(settings);
 return t.innerHTML;
};window.studioPages.data=renderData;
document.addEventListener('change',e=>{if(e.target.id==='column-size-name'){columnSize.selected=e.target.value;document.getElementById('column-size-value').value=columnWidth(columnSize.selected);}});
document.addEventListener('click',e=>{
 const action=e.target.closest('[data-column-size]');if(!action)return;const table=document.querySelector('.scalable-table');if(!table)return;
 if(action.dataset.columnSize==='apply'){const width=Number(document.getElementById('column-size-value').value);if(!Number.isFinite(width)||width<columnSize.min||width>columnSize.max)return toast('Choose a column width from 72 to 1,000 pixels.','error');columnSize.widths.set(columnSize.selected,Math.round(width));}
 if(action.dataset.columnSize==='compact')state.dataset.fields.forEach(f=>columnSize.widths.set(f.name,88));
 if(action.dataset.columnSize==='reset')columnSize.widths.clear();
 if(action.dataset.columnSize==='fit')table.querySelectorAll('th[data-width-column]').forEach(th=>columnSize.widths.set(th.dataset.widthColumn,columnMeasure(table,th.dataset.widthColumn)));
 columnApply(table);
});
document.addEventListener('pointerdown',e=>{
 const handle=e.target.closest('[data-column-grip]');if(!handle)return;e.preventDefault();e.stopImmediatePropagation();
 const name=handle.dataset.columnGrip;columnSize.selected=name;columnSize.drag={name,x:e.clientX,width:columnWidth(name),handle};handle.setPointerCapture(e.pointerId);document.body.classList.add('column-resizing');
},true);
document.addEventListener('pointermove',e=>{const drag=columnSize.drag;if(!drag)return;const table=document.querySelector('.scalable-table');if(!table||!drag.handle.isConnected){columnEndDrag(true);return;}const width=Math.round(Math.max(columnSize.min,Math.min(columnSize.max,drag.width+e.clientX-drag.x)));columnSize.widths.set(drag.name,width);columnApply(table);});
function columnEndDrag(cancel=false){if(cancel&&columnSize.drag){columnSize.widths.set(columnSize.drag.name,columnSize.drag.width);const table=document.querySelector('.scalable-table');if(table)columnApply(table);}columnSize.drag=null;document.body.classList.remove('column-resizing');}
document.addEventListener('pointerup',()=>columnEndDrag());document.addEventListener('pointercancel',()=>columnEndDrag(true));
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&columnSize.drag){columnEndDrag(true);return;}const h=e.target.closest('[data-column-grip]');if(!h||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const name=h.dataset.columnGrip,delta=(e.shiftKey?50:8)*(e.key==='ArrowLeft'?-1:1);columnSize.selected=name;columnSize.widths.set(name,e.key==='Home'?columnSize.min:e.key==='End'?columnSize.max:Math.max(columnSize.min,Math.min(columnSize.max,columnWidth(name)+delta)));columnApply(h.closest('table'));});
document.addEventListener('dblclick',e=>{const h=e.target.closest('[data-column-grip]');if(!h)return;e.preventDefault();columnSize.selected=h.dataset.columnGrip;columnSize.widths.set(h.dataset.columnGrip,columnMeasure(h.closest('table'),h.dataset.columnGrip));columnApply(h.closest('table'));});
const columnSizedRender=render;
render=function(){const host=document.querySelector('.scalable-table')?.closest('.table-scroll'),same=columnSize.instance===state?.dataset?.instance_key,position=host?[host.scrollLeft,host.scrollTop]:null;columnSizedRender();const next=document.querySelector('.scalable-table')?.closest('.table-scroll');if(next&&same&&position){next.scrollLeft=position[0];next.scrollTop=position[1];}};
