function [locs, pks] = peakseek(x, minpeakdist, minpeakh)
% Simple peak detection (replacement for findpeaks without Signal Processing Toolbox)
% Returns peak locations and values
if nargin < 2, minpeakdist = 1; end
if nargin < 3, minpeakh = 0; end

x = x(:)';
dx = diff(x);
locs = find(dx(1:end-1) > 0 & dx(2:end) <= 0) + 1;
if ~isempty(locs)
    pks = x(locs);
    % Filter by minimum height
    keep = pks >= minpeakh;
    locs = locs(keep);
    pks = pks(keep);
    % Filter by minimum distance
    if minpeakdist > 1 && numel(locs) > 1
        keep = true(size(locs));
        for i = 2:numel(locs)
            if locs(i) - locs(find(keep(1:i-1), 1, 'last')) < minpeakdist
                if pks(i) > pks(find(keep(1:i-1), 1, 'last'))
                    keep(find(keep(1:i-1), 1, 'last')) = false;
                else
                    keep(i) = false;
                end
            end
        end
        locs = locs(keep);
        pks = pks(keep);
    end
else
    pks = [];
end
end
