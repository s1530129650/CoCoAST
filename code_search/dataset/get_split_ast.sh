mkdir -p log
java_files_dir=${HOME}/codesearchnet/java_files/all
split_ast_files_dir=${HOME}/codesearchnet/split_ast/all

python get_split_ast.py \
-java_files_dir ${java_files_dir} \
-split_ast_files_dir ${split_ast_files_dir}  \
-output_dir java 2>&1 |tee log/get_split_ast.txt