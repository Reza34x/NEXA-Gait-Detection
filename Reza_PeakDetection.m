fsr = datas(:,37);
[High_pks,High_locs] = findpeaks(fsr, "MinPeakProminence",0.4); %storing the value and location of high peaks
[Low_pks,Low_locs] = findpeaks(-(fsr), "MinPeakProminence",0.4); %storing the value and location of low peaks

figure()
plot(fsr, LineWidth=1)
hold on
plot(High_locs-10, High_pks,  "Marker","v", "MarkerSize", 7, "LineStyle","none", Color='red')
hold on
plot(Low_locs, -Low_pks,  "Marker","^", "MarkerSize", 7, "LineStyle", "none", Color='magenta')
legend("data", "High peaks", "Low peaks")
title("Peak detection")


% plot(fsr, LineWidth=1)
% hold on
% scatter(fsr(High_locs))

ssss















