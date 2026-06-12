import { Header } from '../components/layout/Header';
import { PowerCalculator } from '../components/power/PowerCalculator';

export function PowerAnalysis() {
  return (
    <div>
      <Header
        title="Power Calculator"
        subtitle="Calculate the required sample size and experiment duration before you start"
      />
      <PowerCalculator />
    </div>
  );
}
