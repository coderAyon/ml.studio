// Outliers are visible where students first inspect their CSV, including every table page.
const outlierUI={selected:null,key:''};
const outlierData=renderData;
renderData=function(){
 const t=document.createElement('template');t.innerHTML=outlierData();const d=state.dataset;if(!d)return t.innerHTML;
 const key=d.instance_key+'|'+d.version;if(outlierUI.key!==key){outlierUI.key=key;outlierUI.selected=null;}
 const fields=d.fields.filter(f=>f.outliers?.outlier_count),marks=new Set((d.outlier_marks||[]).map(m=>JSON.stringify([m.row,m.column])));
 const panel=t.content.querySelector('#dataset-search')?.closest('section.panel');
 const summary=document.createElement('section');summary.className='outlier-overview';
 summary.innerHTML=`<div><h2>${d.outlier_cells?`⚑ ${num(d.outlier_cells)} possible outlier values in ${num(d.outlier_rows)} rows`:'No numeric IQR outliers found'}</h2><p>Checked across the full active dataset, including other pages. Flagged cells have an Outlier label. Unusual values need review; they are not automatically errors.</p></div><div class="actions"><button class="button small" data-outlier-filter="${d.outlier_filter?'all':'flagged'}">${d.outlier_filter?'Show all rows':'Show flagged rows'}</button></div>${d.fields.some(f=>f.outliers?.error)?'<p>Some numeric ranges could not be checked reliably; review their column statistics.</p>':''}${fields.length?`<div class="outlier-column-list">${fields.map(f=>`<button data-outlier-column="${esc(f.name)}"><b>${esc(f.name)} · ${f.outliers.outlier_count} flagged</b><small>Typical IQR limits: ${insightNumber(f.outliers.lower)} to ${insightNumber(f.outliers.upper)}</small><small>${f.outlier_examples.map(e=>`Row ${e.row}: ${esc(e.value)}`).join(' · ')}</small></button>`).join('')}</div>`:''}`;
 if(panel)panel.before(summary);else t.content.querySelector('.tabs')?.before(summary);
 if(panel){const headers=Array.from(panel.querySelectorAll('thead th')).map(th=>th.querySelector('[data-sort-column]')?.dataset.sortColumn);panel.querySelectorAll('tbody tr').forEach((tr,i)=>{
  const row=d.preview.positions?.[i]??i+1;const index=tr.querySelector('.index');if(index){index.textContent=row;index.title='Active row position · retained index '+d.preview.index[i];}
  tr.querySelectorAll('td').forEach((td,j)=>{const column=headers[j];if(!column||!marks.has(JSON.stringify([row,column])))return;
   td.classList.add('outlier-cell');td.innerHTML=`<button class="outlier-value" data-outlier-row="${row}" data-outlier-field="${esc(column)}" aria-label="Row ${row}, ${esc(column)}, ${esc(td.textContent)}, possible IQR outlier"><strong>${esc(td.textContent)}</strong><span>⚑ Outlier</span></button>`;
  });
 });}
 const selected=outlierUI.selected;
 if(selected){const f=d.fields.find(f=>f.name===selected.column),r=d.preview.positions?.indexOf(selected.row),c=d.preview.columns.indexOf(selected.column),value=r>=0&&c>=0?d.preview.data[r][c]:selected.value;
  if(f?.outliers){const b=f.outliers,detail=document.createElement('section');detail.className='outlier-detail panel';detail.id='outlier-detail';detail.innerHTML=`<div class="panel-body"><h3>Row ${selected.row} · ${esc(f.name)} · <strong>${esc(value)}</strong></h3><p>This value is outside the IQR limits <b>${insightNumber(b.lower)} to ${insightNumber(b.upper)}</b>. The middle half of values lies between ${insightNumber(b.q1)} and ${insightNumber(b.q3)}.</p><p>Rule: Q1 − 1.5 × IQR / Q3 + 1.5 × IQR. Missing and infinite values are excluded. Scaling alone does not remove outliers.</p><div class="actions"><button class="button" data-outlier-box="${esc(f.name)}">See boxplot</button><button class="button" data-outlier-review="${esc(f.name)}">Review in Prepare data</button></div></div>`;summary.after(detail);}
 }
 return t.innerHTML;
};
window.studioPages.data=renderData;
queryExplorer=function(){cmd('table_query',{search:view.search,sort:explorer.sort,descending:explorer.descending,page:explorer.page,outliers_only:!!state.dataset?.outlier_filter,outlier_column:state.dataset?.outlier_column||''});};
document.addEventListener('click',e=>{
 const cell=e.target.closest('[data-outlier-row]'),column=e.target.closest('[data-outlier-column]'),filter=e.target.closest('[data-outlier-filter]'),box=e.target.closest('[data-outlier-box]'),review=e.target.closest('[data-outlier-review]');
 if(cell){outlierUI.selected={row:Number(cell.dataset.outlierRow),column:cell.dataset.outlierField,value:cell.querySelector('strong').textContent};render();document.getElementById('outlier-detail')?.scrollIntoView({block:'center',behavior:'smooth'});}
 if(filter||column){explorer.page=0;outlierUI.selected=null;cmd('table_query',{search:view.search,sort:explorer.sort,descending:explorer.descending,page:0,outliers_only:column?true:filter.dataset.outlierFilter==='flagged',outlier_column:column?.dataset.outlierColumn||''});}
 if(box){visualsHub.tab='charts';navigate('dataset_visuals');dvSync();datasetVisual.kind='box';datasetVisual.columns=[box.dataset.outlierBox];datasetVisual.show_outliers=true;dvDraw();render();}
 if(review){const issue=state.issues.find(i=>i.title==='Potential outliers'&&i.columns.includes(review.dataset.outlierReview));if(issue){navigate('preprocessing');cmd('preprocess_inspect',{id:issue.id,row:outlierUI.selected?.row});}}
});
