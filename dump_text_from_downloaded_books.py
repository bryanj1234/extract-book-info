from lib_book_corpus import *

print()
print("##############################################################")
print("##############################################################")
print()
print(pathlib.Path(__file__))
print()
print("Running with Python version", sys.version)
print("Running with Python executable", sys.executable)
print()



#########################################################################################################


pdf_file_list = get_book_name_list_for_corpus()

# Get count of books.
num_books = len(pdf_file_list)
print(num_books)
book_num = 0
for pdf_file_str in pdf_file_list:
    book_num += 1

    print("\n\n###", pdf_file_str, "#########################################", "\n")
    print("Book #", str(book_num), "of", str(num_books))

    pickle_dump_font_and_text_information_from_PDF_file(pdf_file_str)



print('DONE')

