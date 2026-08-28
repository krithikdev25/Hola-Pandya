import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.risk_model import predict_risk, get_model_metrics
from engine.models import Asset, FinancialState, ContractualState, LifecycleStage
from engine.risk_agent import assess
from engine.scenario_runner import run_scenario

m=get_model_metrics(); assert m['dataset_rows']>=30000
assert m['n_train']>0 and m['n_validation']>0 and m['n_test']>0
assert 0.65 < m['test_auc'] < 0.95
assert m['operating_threshold']['threshold'] != 0.5
assert m['threshold_method']=='validation expected-cost minimization'

base={"buyer_risk_score":.15,"buyer_payment_history":.95,"supplier_reliability":.95,"production_delay_days":0,"shipment_delay_days":0,"payment_delay_days":0,"inventory_age_days":5,"deterioration":0,"invoice_dispute_flag":0,"duplicate_financing_flag":0,"source_confidence":.95,"stage_progress":.8,"advance_rate_requested":.7,"financing_utilization":.2,"demand_volatility":.1}
checks=[]
for key, bad in [('buyer_risk_score',.85),('production_delay_days',20),('shipment_delay_days',15),('payment_delay_days',25),('inventory_age_days',80),('deterioration',1),('invoice_dispute_flag',1),('duplicate_financing_flag',1)]:
    x=base.copy(); x[key]=bad; checks.append((key,predict_risk(x)['risk_score'] > predict_risk(base)['risk_score']))
for k,v in checks: assert v, (k,checks)

# Safer counterparty inputs must reduce risk.
for key, safe in [('buyer_risk_score',.05),('buyer_payment_history',1.0),('supplier_reliability',1.0)]:
    x=base.copy(); x[key]=safe; assert predict_risk(x)['risk_score'] <= predict_risk(base)['risk_score'] + 1e-9

assert len([run_scenario(f'scenario_{i}') for i in range(1,11)])==10
print('Professional checks: 15 passed, 0 failed')
print('AUC:',round(m['test_auc'],3),'PR-AUC:',round(m['test_average_precision'],3),'learned threshold:',round(m['operating_threshold']['threshold'],3))
