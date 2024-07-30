answer = questdlg('Do you really want to restart MATLAB?', ...
	'Restart Confirmation', ...
	'OK','Cancel','Cancel');
if strcmp(answer,'OK')
    !matlab &
    exit
end