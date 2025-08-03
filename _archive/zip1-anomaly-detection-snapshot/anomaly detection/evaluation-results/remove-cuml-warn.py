linesToWrite = []
with (
    open("./robust-results.txt", "r") as file_in,
    open("./robust-results-1.txt", "w") as file_out,
):
    lines = file_in.readlines()

    for line in lines:
        if "[CUML] [warning]" in line:
            continue
        linesToWrite.append(line)
    file_out.writelines(linesToWrite)
