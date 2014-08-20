function [trimmed] = trimData(dl,tSize)
  trimmed = [];
  labels = unique(dl(:,end));
  for i = 1:length(labels)
    filtered = dl(dl(:,end)==labels(i),:);
    trimmed = [trimmed; datasample(filtered,tSize,'Replace',false)];
  end
  trimmed = trimmed(randperm(size(trimmed,1)),:);
end