def _group_files_by_xml_filename(source, xmls, files):

    grouped_files = {xml: [] for xml in xmls}

    for file in files:

        if file.endswith('.xml'):

            xml_name = file[:-4]  # Assuming XML filenames are the same as file names without the.xml extension

            if xml_name in xmls:

                grouped_files[xml_name].append(file)

    return grouped_files