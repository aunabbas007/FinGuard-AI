const $ = (id) => document.getElementById(id);
let featureNames = [];

async function api(path, options={}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json",...(options.headers||{})},...options});
  const body = await response.json().catch(()=>({detail:"Unexpected server response"}));
  if (!response.ok) throw new Error(body.detail || "Request failed");
  return body;
}
function esc(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
async function loadSchema(){
  try{
    const data=await api("/api/v1/model/schema");
    featureNames=data.features;
    $("feature-help").textContent=`${data.feature_count} numeric inputs loaded from the trained model. Threshold: ${Number(data.threshold).toFixed(4)}.`;
    $("feature-fields").innerHTML=featureNames.map((name,i)=>`<label title="${esc(name)}">${esc(name)}<input type="number" step="any" name="feature_${i}" value="0" required></label>`).join("");
    $("model-status").textContent="READY";$("model-status").style.color="var(--teal)";
    $("model-foot").textContent=`${data.feature_count} features · Random Forest`;
    $("api-status").textContent="API connected";
    $("form-message").textContent="Enter feature values, then run the assessment.";
  }catch(error){
    $("feature-help").textContent="Model schema unavailable. Ensure model artifacts are present on the server.";
    $("feature-fields").innerHTML=`<div class="empty">${esc(error.message)}</div>`;
    $("model-status").textContent="OFFLINE";$("model-foot").textContent="Model artifacts unavailable";
    $("api-status").textContent="API connected · model offline";
  }
}
async function refresh(){
  try{
    const [all,flagged]=await Promise.all([api("/api/v1/transactions?limit=1"),api("/api/v1/transactions?flagged_only=true&limit=1")]);
    $("total-count").textContent=all.total.toLocaleString();
    $("flagged-count").textContent=flagged.total.toLocaleString();
    const data=await api("/api/v1/transactions?limit=20");
    $("transactions-body").innerHTML=data.items.length?data.items.map(t=>{
      const date=new Date(t.created_at).toLocaleString();
      return `<tr><td>${esc(t.external_id||("#"+t.id))}</td><td>${esc(t.currency)} ${Number(t.amount).toLocaleString()}</td><td>${t.risk_score===null?"—":Number(t.risk_score).toFixed(2)+"%"}</td><td><span class="badge ${esc(t.risk_level)}">${esc(t.risk_level)}</span></td><td class="decision">${t.is_flagged?"FLAGGED":"CLEAR"}</td><td>${esc(date)}</td></tr>`;
    }).join(""):`<tr><td colspan="6" class="empty">No assessments yet. Score your first transaction above.</td></tr>`;
  }catch(error){$("transactions-body").innerHTML=`<tr><td colspan="6" class="empty">${esc(error.message)}</td></tr>`;}
}
$("score-form").addEventListener("submit",async event=>{
  event.preventDefault();
  if(!featureNames.length){$("form-message").textContent="Model schema is unavailable.";return;}
  const form=new FormData(event.currentTarget), features={};
  featureNames.forEach((name,i)=>features[name]=Number(form.get("feature_"+i)));
  const payload={external_id:form.get("external_id")||null,amount:Number(form.get("amount")),currency:form.get("currency"),channel:form.get("channel"),sender_account:form.get("sender_account"),receiver_account:form.get("receiver_account"),features};
  $("score-btn").disabled=true;$("score-btn").textContent="Assessing…";$("form-message").textContent="Sending feature vector to model…";
  try{
    const t=await api("/api/v1/transactions/score",{method:"POST",body:JSON.stringify(payload)});
    $("result").classList.remove("hidden");
    $("result").innerHTML=`<div class="result-top"><div><small>MODEL RISK SCORE</small><div class="result-score">${Number(t.risk_score).toFixed(2)}<small>%</small></div></div><span class="badge ${esc(t.risk_level)}">${esc(t.risk_level)}</span></div><p>${esc(t.recommendation)}</p><small>Model output: ${(Number(t.model_probability)*100).toFixed(2)}% · Decision: ${t.is_flagged?"Flagged for human review":"No alert triggered"}</small>`;
    $("form-message").textContent="Assessment saved to transaction history.";
    await refresh();$("result").scrollIntoView({behavior:"smooth",block:"nearest"});
  }catch(error){$("form-message").textContent=error.message;}
  finally{$("score-btn").disabled=false;$("score-btn").innerHTML='Run fraud assessment <span>→</span>';}
});
$("demo-btn").addEventListener("click",async()=>{
  $("form-message").textContent="Loading a random synthetic transaction…";
  try{
    const sample=await api("/api/v1/demo/sample");
    const form=document.getElementById("score-form");
    for(const [key,value] of Object.entries(sample.transaction)){
      const input=form.elements.namedItem(key);
      if(input) input.value=value;
    }
    featureNames.forEach((name,i)=>{
      const input=document.querySelector(`#feature-fields input[name="feature_${i}"]`);
      if(input) input.value=sample.features[name];
    });
    $("form-message").textContent=sample.dataset_notice;
  }catch(error){$("form-message").textContent=error.message;}
});
$("zero-btn").addEventListener("click",()=>document.querySelectorAll("#feature-fields input").forEach(input=>input.value="0"));
$("refresh-btn").addEventListener("click",refresh);
loadSchema();refresh();
